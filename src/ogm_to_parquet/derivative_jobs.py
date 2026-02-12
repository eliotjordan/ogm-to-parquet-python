"""RQ job functions for cloud derivatives processing.

This module contains the job functions that are executed by RQ workers.
Each function processes a single document and must be importable by workers.
"""

import logging
import time
import zipfile
from pathlib import Path

from rq import get_current_job

from .config import IMAGE_FORMATS, VECTOR_FORMATS
from .converters import (
    convert_to_flatgeobuf,
    convert_to_pmtiles,
    convert_to_pyramidal_tiff,
    extract_archive,
)
from .db import DatabaseManager
from .utils import cleanup_scratch, download_file, generate_output_path

logger = logging.getLogger(__name__)


def update_status(
    db_path: str,
    doc_id: str,
    status: str,
    derivative_url: str | None = None,
    error_message: str | None = None,
    max_retries: int = 5,
) -> bool:
    """Update document status in database with retry logic.

    Args:
        db_path: Path to SQLite database
        doc_id: Document ID to update
        status: New status value
        derivative_url: Output file path (for success)
        error_message: Error description (for failure)
        max_retries: Maximum retry attempts for database operations

    Returns:
        True if a row was updated, False if no matching row found
    """
    db = DatabaseManager(db_path)
    extra_fields = {}
    if derivative_url:
        extra_fields["derivative_url"] = derivative_url
    if error_message:
        extra_fields["error_message"] = error_message

    return db.update_status(
        "cloud_derivatives",
        doc_id,
        status,
        extra_fields=extra_fields if extra_fields else None,
        max_retries=max_retries,
    )


def increment_retry_count(db_path: str, doc_id: str, max_retries: int = 5) -> int:
    """Increment and return the retry count for a document.

    Args:
        db_path: Path to SQLite database
        doc_id: Document ID
        max_retries: Maximum retry attempts for database operations

    Returns:
        New retry count after increment
    """
    db = DatabaseManager(db_path)

    # First increment
    db.execute_with_retry(
        "UPDATE cloud_derivatives SET retry_count = COALESCE(retry_count, 0) + 1 WHERE id = ?",
        (doc_id,),
        max_retries=max_retries,
    )

    # Then fetch current value
    results = db.execute_with_retry(
        "SELECT retry_count FROM cloud_derivatives WHERE id = ?",
        (doc_id,),
        max_retries=max_retries,
        commit=False,
    )
    return results[0][0] if results else 0


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

    # Success - update database status
    logger.info(f"[{doc_id}] Conversion complete, updating database status")
    if update_status(db_path, doc_id, "complete", derivative_url=str(final_output)):
        logger.info(f"[{doc_id}] Successfully marked complete: {final_output}")
    else:
        logger.error(f"[{doc_id}] Failed to mark complete in database")
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

    # Success - update database status
    logger.info(f"[{doc_id}] Conversion complete, updating database status")
    if update_status(db_path, doc_id, "complete", derivative_url=str(final_output)):
        logger.info(f"[{doc_id}] Successfully marked complete: {final_output}")
    else:
        logger.error(f"[{doc_id}] Failed to mark complete in database")
    return {"success": True, "output_path": str(final_output)}
