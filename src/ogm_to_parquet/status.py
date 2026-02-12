"""Status reporting for ogm-to-parquet pipeline.

Provides a unified view of the pipeline state across all components:
harvest, derivatives processing, and text extraction.
"""

import argparse
import logging
from pathlib import Path

from .config import DerivativesConfig, HarvestConfig, TextExtractionConfig
from .db import DatabaseManager
from .utils import check_redis

logger = logging.getLogger(__name__)


def get_parquet_status(parquet_path: Path) -> dict:
    """Get status of the harvested Parquet file.

    Args:
        parquet_path: Path to the Parquet file

    Returns:
        Dict with exists flag and document count
    """
    if not parquet_path.exists():
        return {"exists": False, "documents": 0, "size_mb": 0}

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        return {
            "exists": True,
            "documents": table.num_rows,
            "size_mb": round(size_mb, 2),
        }
    except Exception as e:
        logger.warning(f"Could not read Parquet file: {e}")
        return {"exists": True, "documents": -1, "size_mb": 0, "error": str(e)}


def get_derivatives_status(db_path: Path) -> dict:
    """Get status of derivatives processing.

    Args:
        db_path: Path to derivatives database

    Returns:
        Dict with counts by status
    """
    if not db_path.exists():
        return {"exists": False}

    try:
        db = DatabaseManager(db_path)
        if not db.table_exists("cloud_derivatives"):
            return {"exists": True, "table_exists": False}

        # Get counts by status
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, COUNT(*) FROM cloud_derivatives GROUP BY status
            """)
            rows = cursor.fetchall()

        counts = {row[0]: row[1] for row in rows}
        total = sum(counts.values())

        return {
            "exists": True,
            "table_exists": True,
            "total": total,
            "unprocessed": counts.get("unprocessed", 0),
            "enqueued": counts.get("enqueued", 0),
            "in_progress": counts.get("in_progress", 0),
            "complete": counts.get("complete", 0),
            "error": counts.get("error", 0),
        }
    except Exception as e:
        logger.warning(f"Could not read derivatives database: {e}")
        return {"exists": True, "error": str(e)}


def get_text_extraction_status(db_path: Path) -> dict:
    """Get status of text extraction.

    Args:
        db_path: Path to text extraction database

    Returns:
        Dict with counts by status
    """
    if not db_path.exists():
        return {"exists": False}

    try:
        db = DatabaseManager(db_path)
        if not db.table_exists("text_extraction"):
            return {"exists": True, "table_exists": False}

        # Get counts by status
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, COUNT(*) FROM text_extraction GROUP BY status
            """)
            rows = cursor.fetchall()

        counts = {row[0]: row[1] for row in rows}
        total = sum(counts.values())

        return {
            "exists": True,
            "table_exists": True,
            "total": total,
            "unprocessed": counts.get("unprocessed", 0),
            "in_progress": counts.get("in_progress", 0),
            "complete": counts.get("complete", 0),
            "error": counts.get("error", 0),
        }
    except Exception as e:
        logger.warning(f"Could not read text extraction database: {e}")
        return {"exists": True, "error": str(e)}


def get_redis_status(redis_url: str) -> dict:
    """Get Redis queue status.

    Args:
        redis_url: Redis connection URL

    Returns:
        Dict with availability and queue info
    """
    if not check_redis(redis_url):
        return {"available": False}

    try:
        from redis import Redis
        from rq import Queue
        from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

        redis = Redis.from_url(redis_url, socket_connect_timeout=5)
        queue = Queue("derivatives", connection=redis)

        started = StartedJobRegistry(queue=queue)
        finished = FinishedJobRegistry(queue=queue)
        failed = FailedJobRegistry(queue=queue)

        return {
            "available": True,
            "queued": len(queue),
            "running": started.count,
            "finished": finished.count,
            "failed": failed.count,
        }
    except Exception as e:
        logger.warning(f"Could not get Redis queue status: {e}")
        return {"available": True, "error": str(e)}


def get_workflow_status(
    parquet_path: Path | None = None,
    derivatives_db_path: Path | None = None,
    text_extraction_db_path: Path | None = None,
    redis_url: str | None = None,
) -> dict:
    """Get unified status of all pipeline components.

    Args:
        parquet_path: Path to Parquet file (default from HarvestConfig)
        derivatives_db_path: Path to derivatives DB (default from DerivativesConfig)
        text_extraction_db_path: Path to text extraction DB (default from TextExtractionConfig)
        redis_url: Redis URL (default from DerivativesConfig)

    Returns:
        Dict with status of each pipeline component
    """
    harvest_config = HarvestConfig()
    derivatives_config = DerivativesConfig()
    text_config = TextExtractionConfig()

    parquet_path = parquet_path or harvest_config.output_path
    derivatives_db_path = derivatives_db_path or derivatives_config.db_path
    text_extraction_db_path = text_extraction_db_path or text_config.db_path
    redis_url = redis_url or derivatives_config.redis_url

    return {
        "harvest": get_parquet_status(parquet_path),
        "derivatives": get_derivatives_status(derivatives_db_path),
        "text_extraction": get_text_extraction_status(text_extraction_db_path),
        "redis": get_redis_status(redis_url),
    }


def print_status(status: dict) -> None:
    """Print status in a formatted table.

    Args:
        status: Status dict from get_workflow_status()
    """
    print("\nOGM-to-Parquet Pipeline Status")
    print("=" * 50)

    # Harvest status
    harvest = status["harvest"]
    print("\n[Harvest]")
    if harvest.get("exists"):
        print(f"  Parquet file:  exists ({harvest.get('documents', 0):,} documents)")
        print(f"  Size:          {harvest.get('size_mb', 0):.2f} MB")
    else:
        print("  Parquet file:  not found")
        print("  Run: uv run ogm-harvest")

    # Derivatives status
    derivatives = status["derivatives"]
    print("\n[Derivatives]")
    if not derivatives.get("exists"):
        print("  Database:      not found")
        print("  Run: uv run ogm-enrich-prepare")
    elif not derivatives.get("table_exists"):
        print("  Database:      exists (table not created)")
    else:
        total = derivatives.get("total", 0)
        complete = derivatives.get("complete", 0)
        pct = (complete / total * 100) if total > 0 else 0
        print(f"  Total:         {total:,}")
        print(f"  Unprocessed:   {derivatives.get('unprocessed', 0):,}")
        print(f"  Enqueued:      {derivatives.get('enqueued', 0):,}")
        print(f"  In Progress:   {derivatives.get('in_progress', 0):,}")
        print(f"  Complete:      {complete:,} ({pct:.1f}%)")
        print(f"  Errors:        {derivatives.get('error', 0):,}")

    # Text extraction status
    text_ext = status["text_extraction"]
    print("\n[Text Extraction]")
    if not text_ext.get("exists"):
        print("  Database:      not found")
        print("  Run: uv run ogm-enrich-prepare")
    elif not text_ext.get("table_exists"):
        print("  Database:      exists (table not created)")
    else:
        total = text_ext.get("total", 0)
        complete = text_ext.get("complete", 0)
        pct = (complete / total * 100) if total > 0 else 0
        print(f"  Total:         {total:,}")
        print(f"  Unprocessed:   {text_ext.get('unprocessed', 0):,}")
        print(f"  In Progress:   {text_ext.get('in_progress', 0):,}")
        print(f"  Complete:      {complete:,} ({pct:.1f}%)")
        print(f"  Errors:        {text_ext.get('error', 0):,}")

    # Redis status
    redis = status["redis"]
    print("\n[Redis Queue]")
    if not redis.get("available"):
        print("  Status:        not available")
        print("  Start: docker compose up -d")
    else:
        print("  Status:        connected")
        print(f"  Queued:        {redis.get('queued', 0):,}")
        print(f"  Running:       {redis.get('running', 0):,}")
        print(f"  Finished:      {redis.get('finished', 0):,}")
        print(f"  Failed:        {redis.get('failed', 0):,}")

    print("\n" + "=" * 50)


def main():
    """Command-line entry point for status reporting."""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings/errors
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Show status of OGM-to-Parquet pipeline components"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Path to Parquet file (default: ./tmp/ogm.parquet)",
    )
    parser.add_argument(
        "--derivatives-db",
        type=str,
        default=None,
        help="Path to derivatives database (default: ./tmp/derivatives.db)",
    )
    parser.add_argument(
        "--text-extraction-db",
        type=str,
        default=None,
        help="Path to text extraction database (default: ./tmp/text_extraction.db)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output status as JSON instead of formatted table",
    )

    args = parser.parse_args()

    # Convert paths if provided
    parquet_path = Path(args.parquet_path) if args.parquet_path else None
    derivatives_db = Path(args.derivatives_db) if args.derivatives_db else None
    text_extraction_db = Path(args.text_extraction_db) if args.text_extraction_db else None

    status = get_workflow_status(
        parquet_path=parquet_path,
        derivatives_db_path=derivatives_db,
        text_extraction_db_path=text_extraction_db,
        redis_url=args.redis_url,
    )

    if args.json:
        import json

        print(json.dumps(status, indent=2))
    else:
        print_status(status)


if __name__ == "__main__":
    main()
