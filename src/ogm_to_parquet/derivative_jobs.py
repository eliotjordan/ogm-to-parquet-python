"""RQ job functions for cloud derivatives processing.

This module contains the job functions that are executed by RQ workers.
Each function processes a single document and must be importable by workers.
"""

import logging
import os
import shutil
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path

import pyvips
import requests
from rq import get_current_job

logger = logging.getLogger(__name__)

# Vector formats that should be converted to PMTiles via ogr2ogr + tippecanoe
VECTOR_FORMATS = {"GeoJSON", "GeoPackage", "KML", "Shapefile"}

# Image formats that should be converted to pyramidal TIFF via VIPS
IMAGE_FORMATS = {"JPEG", "JPEG2000", "TIFF"}

# File extensions for vector formats
VECTOR_EXTENSIONS = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".gpkg": "GeoPackage",
    ".kml": "KML",
    ".shp": "Shapefile",
}

# File extensions for image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jp2", ".tif", ".tiff"}


def clean_doc_id(doc_id: str) -> str:
    """Remove common prefixes from document IDs.

    Strips prefixes like "ark-85335-", "rutgers-lib:", "stanford-" to get
    the unique part of the identifier for cleaner output paths.

    Args:
        doc_id: Original document ID

    Returns:
        Cleaned document ID with prefix removed
    """
    import re

    # Patterns to strip (prefix patterns that precede unique identifiers)
    prefix_patterns = [
        r"^ark-\d+-",      # ark-85335-xxxxx
        r"^rutgers-lib:",  # rutgers-lib:xxxxx
        r"^stanford-",     # stanford-xxxxx
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
        doc_id: Document ID (must be at least 6 characters)
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


def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Local path to save file
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> Path | None:
    """Extract a zip archive and return path to main data file.

    For shapefiles, returns the .shp file path.
    For other archives, returns the first recognized data file.

    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to

    Returns:
        Path to the main data file, or None if not found
    """
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)

        # Look for shapefile first
        shp_files = list(extract_dir.rglob("*.shp"))
        if shp_files:
            return shp_files[0]

        # Look for other vector formats
        for ext in VECTOR_EXTENSIONS:
            files = list(extract_dir.rglob(f"*{ext}"))
            if files:
                return files[0]

        # Look for image files
        for ext in IMAGE_EXTENSIONS:
            files = list(extract_dir.rglob(f"*{ext}"))
            if files:
                return files[0]

        logger.warning(f"No recognized data file found in archive {archive_path}")
        return None
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file {archive_path}: {e}")
        return None


def convert_to_flatgeobuf(input_path: Path, output_path: Path) -> bool:
    """Convert vector data to FlatGeobuf using ogr2ogr.

    Reprojects to EPSG:4326 and handles partial reprojection for edge cases.

    Args:
        input_path: Path to input vector file
        output_path: Path for output FlatGeobuf file

    Returns:
        True if successful, False otherwise
    """
    env = os.environ.copy()
    env["OGR_ENABLE_PARTIAL_REPROJECTION"] = "YES"

    cmd = [
        "ogr2ogr",
        "-f",
        "FlatGeobuf",
        "-t_srs",
        "EPSG:4326",
        "-skipfailures",
        "-preserve_fid",
        "-makevalid",
        str(output_path),
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"ogr2ogr failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ogr2ogr timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"ogr2ogr error: {e}")
        return False


def convert_to_pmtiles(
    input_path: Path, output_path: Path, timeout: int = 1800
) -> bool:
    """Convert FlatGeobuf to PMTiles using tippecanoe.

    First tries with -zg flag for automatic zoom detection.
    If that fails, retries without -zg (needed for single-feature layers).

    Args:
        input_path: Path to input FlatGeobuf file
        output_path: Path for output PMTiles file
        timeout: Timeout in seconds for tippecanoe

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Primary command with -zg
    cmd_with_zg = [
        "tippecanoe",
        "--force",
        "--maximum-tile-features=10000",
        "--no-tile-size-limit",
        "-zg",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "-o",
        str(output_path),
        str(input_path),
    ]

    try:
        result = subprocess.run(
            cmd_with_zg, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return True

        logger.warning(f"tippecanoe with -zg failed, retrying without: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"tippecanoe timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"tippecanoe error: {e}")
        return False

    # Fallback command without -zg (for single-feature layers)
    cmd_no_zg = [
        "tippecanoe",
        "--force",
        "--maximum-tile-features=10000",
        "--no-tile-size-limit",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "-o",
        str(output_path),
        str(input_path),
    ]

    try:
        result = subprocess.run(
            cmd_no_zg, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.error(f"tippecanoe fallback failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"tippecanoe fallback timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"tippecanoe fallback error: {e}")
        return False


def convert_to_pyramidal_tiff(input_path: Path, output_path: Path) -> bool:
    """Convert image to pyramidal TIFF using VIPS.

    Uses JPEG compression with pyramidal tiles for efficient viewing.

    Args:
        input_path: Path to input image file
        output_path: Path for output TIFF file

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = pyvips.Image.new_from_file(str(input_path))
        image.tiffsave(
            str(output_path),
            compression="jpeg",
            pyramid=True,
            tile=True,
            Q=90,  # quality
            tile_width=1024,
            tile_height=1024,
            strip=True,
        )
        return True
    except pyvips.Error as e:
        logger.error(f"VIPS conversion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"VIPS error: {e}")
        return False


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with WAL mode and busy timeout.

    Uses WAL mode for better concurrent access and sets a busy timeout
    to wait for locks instead of failing immediately.

    Args:
        db_path: Path to SQLite database

    Returns:
        Database connection configured for concurrent access
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
    return conn


def update_status(
    db_path: str,
    doc_id: str,
    status: str,
    derivative_url: str | None = None,
    error_message: str | None = None,
    max_retries: int = 5,
) -> None:
    """Update document status in database with retry logic.

    Args:
        db_path: Path to SQLite database
        doc_id: Document ID to update
        status: New status value
        derivative_url: Output file path (for success)
        error_message: Error description (for failure)
        max_retries: Maximum retry attempts for database operations
    """
    for attempt in range(max_retries):
        try:
            conn = get_db_connection(db_path)
            try:
                cursor = conn.cursor()
                if derivative_url:
                    cursor.execute(
                        "UPDATE cloud_derivatives SET status = ?, derivative_url = ? WHERE id = ?",
                        (status, derivative_url, doc_id),
                    )
                elif error_message:
                    cursor.execute(
                        "UPDATE cloud_derivatives SET status = ?, error_message = ? WHERE id = ?",
                        (status, error_message, doc_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE cloud_derivatives SET status = ? WHERE id = ?",
                        (status, doc_id),
                    )
                conn.commit()
                return
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 0.1  # Exponential backoff
                logger.warning(f"Database locked, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                raise


def increment_retry_count(db_path: str, doc_id: str, max_retries: int = 5) -> int:
    """Increment and return the retry count for a document.

    Args:
        db_path: Path to SQLite database
        doc_id: Document ID
        max_retries: Maximum retry attempts for database operations

    Returns:
        New retry count after increment
    """
    for attempt in range(max_retries):
        try:
            conn = get_db_connection(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE cloud_derivatives SET retry_count = COALESCE(retry_count, 0) + 1 WHERE id = ?",
                    (doc_id,),
                )
                conn.commit()
                cursor.execute("SELECT retry_count FROM cloud_derivatives WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 0.1
                logger.warning(f"Database locked, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                raise
    return 0


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


def handle_job_failure(
    db_path: str,
    doc_id: str,
    error_message: str,
) -> None:
    """Handle job failure with appropriate status based on retry state.

    If retries remain, marks as 'enqueued' so the job can be retried.
    If no retries remain, marks as 'error'.

    Args:
        db_path: Path to SQLite database
        doc_id: Document ID
        error_message: Error description
    """
    job = get_current_job()

    # Check if retries remain
    has_retries = False
    if job and job.retries_left is not None and job.retries_left > 0:
        has_retries = True

    if has_retries:
        # Will be retried - mark as enqueued
        logger.info(f"[{doc_id}] Failed, {job.retries_left} retries remaining. Re-enqueueing.")
        update_status(db_path, doc_id, "enqueued")
    else:
        # No retries left - mark as error
        logger.error(f"[{doc_id}] Failed permanently: {error_message}")
        update_status(db_path, doc_id, "error", error_message=error_message)


def process_derivative(
    doc_id: str,
    download_url: str,
    doc_format: str,
    db_path: str,
    scratch_dir: str,
    output_dir: str,
    tippecanoe_timeout: int = 1800,
    max_retries: int = 5,
    download_delay: float = 0.0,
) -> dict:
    """Process a single derivative document.

    This is the main job function executed by RQ workers.

    For vector formats: Downloads, converts to FlatGeobuf, then to PMTiles.
    For image formats: Downloads, converts to pyramidal TIFF.

    Args:
        doc_id: Document ID
        download_url: URL to download the source file
        doc_format: Format string (e.g., "GeoJSON", "Shapefile", "JPEG")
        db_path: Path to SQLite database
        scratch_dir: Temporary directory for processing
        output_dir: Output directory for derivatives
        tippecanoe_timeout: Timeout for tippecanoe in seconds
        max_retries: Maximum retry attempts
        download_delay: Delay in seconds before downloading (rate limiting)

    Returns:
        Dict with "success" bool and either "output_path" or "error" message
    """
    scratch_path = Path(scratch_dir)
    output_path = Path(output_dir)
    doc_scratch = scratch_path / doc_id

    try:
        # Mark as in_progress
        update_status(db_path, doc_id, "in_progress")

        # Determine processing type
        is_vector = doc_format in VECTOR_FORMATS
        is_image = doc_format in IMAGE_FORMATS

        if not is_vector and not is_image:
            error_msg = f"Unsupported format: {doc_format}"
            update_status(db_path, doc_id, "error", error_message=error_msg)
            return {"success": False, "error": error_msg}

        # Create scratch directory for this document
        doc_scratch.mkdir(parents=True, exist_ok=True)

        # Determine download filename from URL
        url_path = download_url.split("?")[0]  # Remove query params
        url_filename = url_path.split("/")[-1]
        if not url_filename or "." not in url_filename:
            # Fallback based on format
            if is_vector:
                url_filename = f"data.{'zip' if doc_format == 'Shapefile' else 'geojson'}"
            else:
                url_filename = "image.tif"

        download_path = doc_scratch / url_filename

        # Rate limiting delay before download
        if download_delay > 0:
            time.sleep(download_delay)

        # Download the file
        logger.info(f"[{doc_id}] Downloading from {download_url}")
        if not download_file(download_url, download_path):
            error_msg = f"Failed to download from {download_url}"
            handle_job_failure(db_path, doc_id, error_msg)
            return {"success": False, "error": error_msg}

        # Handle archives (zip files)
        input_path = download_path
        if zipfile.is_zipfile(download_path) or download_path.suffix.lower() == ".zip":
            extract_dir = doc_scratch / "extracted"
            extracted = extract_archive(download_path, extract_dir)
            if not extracted:
                error_msg = "Failed to extract archive or find data file"
                handle_job_failure(db_path, doc_id, error_msg)
                return {"success": False, "error": error_msg}
            input_path = extracted

        # Process based on format type
        if is_vector:
            return _process_vector(
                doc_id, input_path, db_path, doc_scratch, output_path, tippecanoe_timeout
            )
        else:
            return _process_image(doc_id, input_path, db_path, output_path)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(f"[{doc_id}] {error_msg}")
        handle_job_failure(db_path, doc_id, error_msg)
        return {"success": False, "error": error_msg}

    finally:
        # Always cleanup scratch files
        cleanup_scratch(scratch_path, doc_id)


def _process_vector(
    doc_id: str,
    input_path: Path,
    db_path: str,
    scratch_dir: Path,
    output_dir: Path,
    tippecanoe_timeout: int,
) -> dict:
    """Process a vector file to PMTiles.

    Args:
        doc_id: Document ID
        input_path: Path to input vector file
        db_path: Path to SQLite database
        scratch_dir: Document's scratch directory
        output_dir: Base output directory
        tippecanoe_timeout: Timeout for tippecanoe

    Returns:
        Dict with "success" bool and either "output_path" or "error" message
    """
    # Convert to FlatGeobuf
    fgb_path = scratch_dir / "data.fgb"
    logger.info(f"[{doc_id}] Converting to FlatGeobuf")
    if not convert_to_flatgeobuf(input_path, fgb_path):
        error_msg = "Failed to convert to FlatGeobuf"
        handle_job_failure(db_path, doc_id, error_msg)
        return {"success": False, "error": error_msg}

    # Convert to PMTiles
    final_output = generate_output_path(output_dir, doc_id, ".pmtiles")
    logger.info(f"[{doc_id}] Converting to PMTiles")
    if not convert_to_pmtiles(fgb_path, final_output, tippecanoe_timeout):
        error_msg = "Failed to convert to PMTiles"
        handle_job_failure(db_path, doc_id, error_msg)
        return {"success": False, "error": error_msg}

    # Success
    logger.info(f"[{doc_id}] Complete: {final_output}")
    update_status(db_path, doc_id, "complete", derivative_url=str(final_output))
    return {"success": True, "output_path": str(final_output)}


def _process_image(
    doc_id: str,
    input_path: Path,
    db_path: str,
    output_dir: Path,
) -> dict:
    """Process an image file to pyramidal TIFF.

    Args:
        doc_id: Document ID
        input_path: Path to input image file
        db_path: Path to SQLite database
        output_dir: Base output directory

    Returns:
        Dict with "success" bool and either "output_path" or "error" message
    """
    final_output = generate_output_path(output_dir, doc_id, ".tif")
    logger.info(f"[{doc_id}] Converting to pyramidal TIFF")
    if not convert_to_pyramidal_tiff(input_path, final_output):
        error_msg = "Failed to convert to pyramidal TIFF"
        handle_job_failure(db_path, doc_id, error_msg)
        return {"success": False, "error": error_msg}

    # Success
    logger.info(f"[{doc_id}] Complete: {final_output}")
    update_status(db_path, doc_id, "complete", derivative_url=str(final_output))
    return {"success": True, "output_path": str(final_output)}
