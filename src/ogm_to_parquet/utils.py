"""Common utilities for ogm-to-parquet.

Provides shared functions for logging setup, directory management,
file downloads, prerequisite checking, and ID manipulation.
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def setup_logging(
    name: str = "ogm_to_parquet",
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Log message format

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=level, format=format_string)
    return logging.getLogger(name)


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = 300,
    headers: dict[str, str] | None = None,
) -> bool:
    """Download a file from URL to destination path.

    Uses streaming to handle large files efficiently.

    Args:
        url: URL to download from
        dest_path: Local path to save file
        timeout: Request timeout in seconds
        headers: Optional custom headers

    Returns:
        True if successful, False otherwise
    """
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.get(url, headers=default_headers, timeout=timeout, stream=True)
        response.raise_for_status()

        ensure_directory(dest_path.parent)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def check_redis(redis_url: str = "redis://localhost:6379") -> bool:
    """Check if Redis is available.

    Args:
        redis_url: Redis connection URL

    Returns:
        True if connected, False otherwise
    """
    try:
        from redis import Redis

        redis = Redis.from_url(redis_url, socket_connect_timeout=5)
        redis.ping()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
        return False


def check_tool_available(tool_name: str) -> bool:
    """Check if an external tool is available in PATH.

    Args:
        tool_name: Name of the tool (e.g., 'ogr2ogr', 'tippecanoe')

    Returns:
        True if tool is available, False otherwise
    """
    try:
        subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_prerequisites(
    checks: list[str], redis_url: str = "redis://localhost:6379"
) -> dict[str, bool]:
    """Validate availability of required prerequisites.

    Args:
        checks: List of prerequisite names to check.
                Supported: 'redis', 'ogr2ogr', 'tippecanoe', 'ollama'
        redis_url: Redis URL for redis check

    Returns:
        Dictionary mapping check name to availability status
    """
    results = {}

    for check in checks:
        if check == "redis":
            results["redis"] = check_redis(redis_url)
        elif check == "ollama":
            results["ollama"] = check_ollama()
        elif check in ("ogr2ogr", "tippecanoe"):
            results[check] = check_tool_available(check)
        else:
            logger.warning(f"Unknown prerequisite check: {check}")
            results[check] = False

    return results


def check_ollama(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running.

    Args:
        ollama_url: Ollama API base URL

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {ollama_url}: {e}")
        return False


def clean_doc_id(doc_id: str) -> str:
    """Remove common prefixes from document IDs.

    Strips prefixes like "ark-85335-", "rutgers-lib:", "stanford-" to get
    the unique part of the identifier for cleaner output paths.

    Args:
        doc_id: Original document ID

    Returns:
        Cleaned document ID with prefix removed
    """
    # Patterns to strip (prefix patterns that precede unique identifiers)
    prefix_patterns = [
        r"^ark-\d+-",  # ark-85335-xxxxx
        r"^rutgers-lib:",  # rutgers-lib:xxxxx
        r"^stanford-",  # stanford-xxxxx
    ]

    cleaned = doc_id
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    return cleaned


def generate_output_path(output_dir: Path, doc_id: str, extension: str) -> Path:
    """Generate nested output path from document ID.

    For ID "060c078970f84019a0c0426016c9d151", creates:
    output_dir/06/0c/07/060c078970f84019a0c0426016c9d151/dataset{extension}

    Common prefixes (ark-NNNNN-, rutgers-lib:, stanford-) are stripped
    to create cleaner paths.

    Args:
        output_dir: Base output directory
        doc_id: Document ID (must be at least 6 characters after cleaning)
        extension: File extension including dot (e.g., ".pmtiles", ".tif")

    Returns:
        Full path to output file
    """
    # Clean the ID by removing common prefixes
    clean_id = clean_doc_id(doc_id)

    if len(clean_id) < 6:
        # Pad short IDs with zeros
        clean_id = clean_id.zfill(6)

    return (
        output_dir
        / clean_id[0:2]
        / clean_id[2:4]
        / clean_id[4:6]
        / clean_id
        / f"dataset{extension}"
    )


def cleanup_scratch(scratch_dir: Path, doc_id: str) -> None:
    """Clean up temporary files for a document.

    Args:
        scratch_dir: Base scratch directory
        doc_id: Document ID (used as subdirectory name)
    """
    doc_scratch = scratch_dir / doc_id
    if doc_scratch.exists():
        try:
            shutil.rmtree(doc_scratch)
        except Exception as e:
            logger.warning(f"Failed to cleanup {doc_scratch}: {e}")
