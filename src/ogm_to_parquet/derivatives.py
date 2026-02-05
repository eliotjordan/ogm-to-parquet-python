"""Cloud derivatives processing pipeline using RQ.

This module provides the main interface for processing geospatial data from
the cloud_derivatives table, converting vector files to PMTiles and images
to pyramidal TIFFs.
"""

import argparse
import logging
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

from redis import Redis
from rq import Queue, Retry
from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

from ogm_to_parquet.derivative_jobs import process_derivative

logger = logging.getLogger(__name__)

# Queue name for derivative processing
QUEUE_NAME = "derivatives"

# Default settings
DEFAULT_DB_PATH = "./tmp/enrichment.db"
DEFAULT_SCRATCH_DIR = "./tmp/scratch"
DEFAULT_OUTPUT_DIR = "./tmp/cloud_derivatives"
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_TIPPECANOE_TIMEOUT = 1800  # 30 minutes
DEFAULT_MAX_RETRIES = 5
DEFAULT_WORKERS = 5
DEFAULT_DELAY = 1.0  # seconds between job submissions


class DerivativeProcessor:
    """Orchestrates cloud derivative processing via RQ job queue.

    This class handles:
    - Loading documents from the database
    - Enqueueing jobs to RQ workers
    - Managing worker processes

    Use rq-dashboard for monitoring: rq-dashboard --redis-url redis://localhost:6379
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        scratch_dir: str = DEFAULT_SCRATCH_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        redis_url: str = DEFAULT_REDIS_URL,
        tippecanoe_timeout: int = DEFAULT_TIPPECANOE_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        delay: float = DEFAULT_DELAY,
    ):
        """Initialize the derivative processor.

        Args:
            db_path: Path to SQLite database with cloud_derivatives table
            scratch_dir: Directory for temporary processing files
            output_dir: Directory for output derivative files
            redis_url: Redis connection URL
            tippecanoe_timeout: Timeout in seconds for tippecanoe processing
            max_retries: Maximum retry attempts for failed jobs
            delay: Delay in seconds between job submissions (rate limiting)
        """
        # Convert all paths to absolute to ensure workers can find them
        self.db_path = Path(db_path).resolve()
        self.scratch_dir = Path(scratch_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.redis_url = redis_url
        self.tippecanoe_timeout = tippecanoe_timeout
        self.max_retries = max_retries
        self.delay = delay

        # Ensure directories exist
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy Redis connection (initialized on first use)
        self._redis: Redis | None = None
        self._queue: Queue | None = None

    @property
    def redis(self) -> Redis:
        """Lazy Redis connection with timeout."""
        if self._redis is None:
            logger.debug(f"Connecting to Redis at {self.redis_url}")
            self._redis = Redis.from_url(
                self.redis_url,
                socket_connect_timeout=5,
                socket_timeout=10,
            )
            # Test the connection
            self._redis.ping()
            logger.debug("Redis connection established")
        return self._redis

    @property
    def queue(self) -> Queue:
        """Lazy queue initialization."""
        if self._queue is None:
            logger.debug(f"Initializing queue '{QUEUE_NAME}'")
            self._queue = Queue(QUEUE_NAME, connection=self.redis)
        return self._queue

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode for concurrent access."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def ensure_retry_column(self) -> None:
        """Add retry_count column to database if it doesn't exist."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(cloud_derivatives)")
            columns = [row[1] for row in cursor.fetchall()]
            if "retry_count" not in columns:
                cursor.execute(
                    "ALTER TABLE cloud_derivatives ADD COLUMN retry_count INTEGER DEFAULT 0"
                )
                conn.commit()
                logger.info("Added retry_count column to cloud_derivatives table")
        finally:
            conn.close()

    def get_pending_documents(self, doc_id: str | None = None) -> list[dict]:
        """Get documents that need processing.

        Args:
            doc_id: Optional specific document ID to process

        Returns:
            List of document dicts with id, download_url, format
        """
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()

            if doc_id:
                # Process specific document regardless of status
                cursor.execute(
                    """
                    SELECT id, download_url, format
                    FROM cloud_derivatives
                    WHERE id = ?
                    """,
                    (doc_id,),
                )
            else:
                # Get unprocessed documents in random order
                cursor.execute(
                    """
                    SELECT id, download_url, format
                    FROM cloud_derivatives
                    WHERE status = 'unprocessed'
                    ORDER BY RANDOM()
                    """
                )

            rows = cursor.fetchall()
            return [
                {"id": row[0], "download_url": row[1], "format": row[2]}
                for row in rows
            ]
        finally:
            conn.close()

    def _mark_enqueued(self, doc_ids: list[str]) -> None:
        """Mark documents as enqueued in the database.

        Args:
            doc_ids: List of document IDs to mark as enqueued
        """
        if not doc_ids:
            return

        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            # Batch update for efficiency
            placeholders = ",".join("?" * len(doc_ids))
            cursor.execute(
                f"UPDATE cloud_derivatives SET status = 'enqueued' WHERE id IN ({placeholders})",
                doc_ids,
            )
            conn.commit()
        finally:
            conn.close()

    def enqueue_documents(self, doc_id: str | None = None) -> int:
        """Enqueue documents for processing.

        Args:
            doc_id: Optional specific document ID to process

        Returns:
            Number of jobs enqueued
        """
        documents = self.get_pending_documents(doc_id)

        if not documents:
            logger.info("No documents to process")
            return 0

        total = len(documents)
        logger.info(f"Enqueueing {total} documents for processing")

        # Test Redis connection before starting
        logger.info("Connecting to Redis...")
        try:
            _ = self.queue  # Trigger lazy connection
            logger.info("Redis connection ready")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return 0

        # Exponential backoff intervals for retries (in seconds)
        retry_intervals = [60, 120, 240, 480, 960]

        enqueued = 0
        errors = 0
        enqueued_ids: list[str] = []
        print(f"Enqueuing: 0/{total} (0.0%)", end="", flush=True)
        for i, doc in enumerate(documents, 1):
            try:
                # Total job timeout: tippecanoe timeout + buffer for download/conversion
                job_timeout = self.tippecanoe_timeout + 900  # 15 min buffer

                # Sanitize job ID (RQ doesn't allow colons in job IDs)
                safe_job_id = f"derivative_{doc['id'].replace(':', '_')}"

                self.queue.enqueue(
                    process_derivative,
                    args=(
                        doc["id"],
                        doc["download_url"],
                        doc["format"],
                        str(self.db_path),
                        str(self.scratch_dir),
                        str(self.output_dir),
                        self.tippecanoe_timeout,
                        self.max_retries,
                        self.delay,  # Rate limiting delay applied before download
                    ),
                    job_timeout=job_timeout,
                    retry=Retry(max=self.max_retries, interval=retry_intervals),
                    job_id=safe_job_id,
                )
                enqueued += 1
                enqueued_ids.append(doc["id"])

                # Batch update database every 500 documents
                if len(enqueued_ids) >= 500:
                    self._mark_enqueued(enqueued_ids)
                    enqueued_ids = []

                # Progress logging - more frequent at start, then every 100
                if enqueued <= 10 or enqueued % 100 == 0 or enqueued == total:
                    pct = (i / total) * 100
                    print(f"\rEnqueuing: {enqueued}/{total} ({pct:.1f}%)", end="", flush=True)

            except Exception as e:
                errors += 1
                logger.error(f"Failed to enqueue {doc['id']}: {e}")

        # Mark remaining enqueued documents
        if enqueued_ids:
            self._mark_enqueued(enqueued_ids)

        print()  # Newline after progress
        logger.info(f"Enqueued {enqueued} jobs ({errors} errors)")
        return enqueued

    def get_queue_stats(self) -> dict:
        """Get current queue statistics.

        Returns:
            Dict with queued, started, finished, failed counts
        """
        started_registry = StartedJobRegistry(queue=self.queue)
        finished_registry = FinishedJobRegistry(queue=self.queue)
        failed_registry = FailedJobRegistry(queue=self.queue)

        return {
            "queued": len(self.queue),
            "started": started_registry.count,
            "finished": finished_registry.count,
            "failed": failed_registry.count,
        }

    def start_workers(self, count: int = DEFAULT_WORKERS) -> list[subprocess.Popen]:
        """Start RQ worker processes.

        Args:
            count: Number of worker processes to spawn

        Returns:
            List of Popen objects for worker processes
        """
        workers = []

        # Set environment for workers - fixes macOS fork() crash with pyvips/Objective-C
        worker_env = os.environ.copy()
        worker_env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        for i in range(count):
            worker_name = f"derivative-worker-{i}"
            cmd = [
                sys.executable,
                "-m",
                "rq.cli",
                "worker",
                "--url",
                self.redis_url,
                "--name",
                worker_name,
                QUEUE_NAME,
            ]

            proc = subprocess.Popen(cmd, env=worker_env)
            workers.append(proc)
            logger.info(f"Started worker {worker_name} (PID: {proc.pid})")

        return workers

    def stop_workers(self, workers: list[subprocess.Popen], timeout: int = 30) -> None:
        """Stop worker processes gracefully.

        Args:
            workers: List of Popen objects
            timeout: Seconds to wait before force killing
        """
        for proc in workers:
            proc.send_signal(signal.SIGTERM)

        # Wait for workers to finish current jobs
        for proc in workers:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Worker {proc.pid} didn't stop, killing")
                proc.kill()

    def process_all(
        self, doc_id: str | None = None, workers: int = DEFAULT_WORKERS
    ) -> list[subprocess.Popen]:
        """Run the full processing pipeline.

        Enqueues documents and starts workers. Workers run until interrupted.
        Use rq-dashboard for monitoring: rq-dashboard --redis-url redis://localhost:6379

        Args:
            doc_id: Optional specific document ID to process
            workers: Number of worker processes

        Returns:
            List of worker Popen objects (caller should handle shutdown)
        """
        # Ensure database schema is up to date
        self.ensure_retry_column()

        # Enqueue any new jobs
        enqueued = self.enqueue_documents(doc_id)

        # Check if there are jobs in the queue (including previously enqueued)
        stats = self.get_queue_stats()
        total_pending = stats["queued"] + stats["started"]

        if enqueued == 0 and total_pending == 0:
            logger.info("No jobs to process")
            return []

        if total_pending > 0:
            logger.info(f"Found {total_pending} jobs in queue")

        # Start workers
        logger.info(f"Starting {workers} workers. Use rq-dashboard to monitor progress.")
        logger.info("Press Ctrl+C to stop workers.")
        return self.start_workers(workers)


def check_redis_connection(redis_url: str) -> bool:
    """Check if Redis is available.

    Args:
        redis_url: Redis connection URL

    Returns:
        True if connected, False otherwise
    """
    try:
        redis = Redis.from_url(redis_url)
        redis.ping()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
        return False


def check_tools_available() -> bool:
    """Check if required external tools are installed.

    Returns:
        True if all tools available, False otherwise
    """
    tools = ["ogr2ogr", "tippecanoe"]
    missing = []

    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)

    if missing:
        logger.error(f"Missing required tools: {', '.join(missing)}")
        return False

    return True


def main():
    """Command-line entry point for derivative processing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Process cloud derivatives to PMTiles and pyramidal TIFFs"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=DEFAULT_SCRATCH_DIR,
        help=f"Directory for temporary files (default: {DEFAULT_SCRATCH_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=DEFAULT_REDIS_URL,
        help=f"Redis connection URL (default: {DEFAULT_REDIS_URL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--tippecanoe-timeout",
        type=int,
        default=DEFAULT_TIPPECANOE_TIMEOUT,
        help=f"Timeout for tippecanoe in seconds (default: {DEFAULT_TIPPECANOE_TIMEOUT})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between job submissions in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Process only this document ID",
    )
    parser.add_argument(
        "--enqueue-only",
        action="store_true",
        help="Only enqueue jobs, don't start workers or monitor",
    )
    parser.add_argument(
        "--workers-only",
        action="store_true",
        help="Start workers for existing queue without enqueueing new jobs",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print queue statistics and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview documents to process without queuing or processing",
    )

    args = parser.parse_args()

    # Check Redis connection (not needed for dry-run)
    if not args.dry_run and not check_redis_connection(args.redis_url):
        logger.error("Cannot proceed without Redis connection")
        sys.exit(1)

    # Check external tools (unless just stats/dry-run)
    if not args.stats and not args.dry_run:
        if not check_tools_available():
            logger.error("Cannot proceed without required tools")
            sys.exit(1)

    processor = DerivativeProcessor(
        db_path=args.db_path,
        scratch_dir=args.scratch_dir,
        output_dir=args.output_dir,
        redis_url=args.redis_url,
        tippecanoe_timeout=args.tippecanoe_timeout,
        max_retries=args.max_retries,
        delay=args.delay,
    )

    try:
        if args.dry_run:
            # Preview documents without processing
            docs = processor.get_pending_documents(args.id)
            if not docs:
                print("No documents to process")
            else:
                print(f"Would process {len(docs)} document(s):\n")
                # Group by format
                by_format: dict[str, list] = {}
                for doc in docs:
                    fmt = doc["format"]
                    if fmt not in by_format:
                        by_format[fmt] = []
                    by_format[fmt].append(doc)

                for fmt, fmt_docs in sorted(by_format.items()):
                    print(f"{fmt} ({len(fmt_docs)}):")
                    for doc in fmt_docs[:5]:  # Show first 5 of each format
                        print(f"  - {doc['id']}: {doc['download_url'][:60]}...")
                    if len(fmt_docs) > 5:
                        print(f"  ... and {len(fmt_docs) - 5} more")
                    print()

        elif args.stats:
            # Just print stats and exit
            stats = processor.get_queue_stats()
            print("Queue Statistics:")
            print(f"  Queued:   {stats['queued']}")
            print(f"  Running:  {stats['started']}")
            print(f"  Finished: {stats['finished']}")
            print(f"  Failed:   {stats['failed']}")
            print("\nUse rq-dashboard for real-time monitoring:")
            print("  rq-dashboard --redis-url redis://localhost:6379")

        elif args.enqueue_only:
            # Just enqueue, don't start workers
            processor.ensure_retry_column()
            enqueued = processor.enqueue_documents(args.id)
            print(f"Enqueued {enqueued} jobs")
            print("\nStart workers with: uv run ogm-enrich-derivatives --workers-only")
            print("Monitor with: rq-dashboard --redis-url redis://localhost:6379")

        elif args.workers_only:
            # Start workers for existing queue without enqueueing
            stats = processor.get_queue_stats()
            total_pending = stats["queued"] + stats["started"]
            if total_pending == 0:
                print("No jobs in queue")
            else:
                print(f"Starting {args.workers} workers for {total_pending} queued jobs")
                print("Monitor with: rq-dashboard --redis-url redis://localhost:6379")
                print("Press Ctrl+C to stop workers.\n")
                worker_procs = processor.start_workers(args.workers)
                try:
                    # Wait for interrupt
                    for proc in worker_procs:
                        proc.wait()
                except KeyboardInterrupt:
                    print("\nStopping workers...")
                    processor.stop_workers(worker_procs)
                print(f"Final stats: {processor.get_queue_stats()}")

        else:
            # Full processing: enqueue + workers
            print("Monitor with: rq-dashboard --redis-url redis://localhost:6379")
            worker_procs = processor.process_all(doc_id=args.id, workers=args.workers)
            if worker_procs:
                try:
                    # Wait for interrupt
                    for proc in worker_procs:
                        proc.wait()
                except KeyboardInterrupt:
                    print("\nStopping workers...")
                    processor.stop_workers(worker_procs)
                print(f"Final stats: {processor.get_queue_stats()}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)


if __name__ == "__main__":
    main()
