"""Format conversion utilities for cloud derivatives.

Provides functions for converting vector files to PMTiles via ogr2ogr + tippecanoe,
and image files to pyramidal TIFFs via pyvips.
"""

import logging
import os
import subprocess
import zipfile
from pathlib import Path

import pyvips

from .config import IMAGE_EXTENSIONS, VECTOR_EXTENSIONS

logger = logging.getLogger(__name__)


def convert_to_flatgeobuf(
    input_path: Path,
    output_path: Path,
    timeout: int = 600,
) -> bool:
    """Convert vector data to FlatGeobuf using ogr2ogr.

    Reprojects to EPSG:4326 and handles partial reprojection for edge cases.

    Args:
        input_path: Path to input vector file
        output_path: Path for output FlatGeobuf file
        timeout: Timeout in seconds for ogr2ogr (default: 10 minutes)

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
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"ogr2ogr failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"ogr2ogr timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"ogr2ogr error: {e}")
        return False


def convert_to_pmtiles(
    input_path: Path,
    output_path: Path,
    timeout: int = 1800,
) -> bool:
    """Convert FlatGeobuf to PMTiles using tippecanoe.

    First tries with -zg flag for automatic zoom detection.
    If that fails, retries without -zg (needed for single-feature layers).

    Args:
        input_path: Path to input FlatGeobuf file
        output_path: Path for output PMTiles file
        timeout: Timeout in seconds for tippecanoe (default: 30 minutes)

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
        result = subprocess.run(cmd_with_zg, capture_output=True, text=True, timeout=timeout)
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
        result = subprocess.run(cmd_no_zg, capture_output=True, text=True, timeout=timeout)
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


def convert_to_pyramidal_tiff(
    input_path: Path,
    output_path: Path,
    quality: int = 90,
    tile_size: int = 1024,
) -> bool:
    """Convert image to pyramidal TIFF using VIPS.

    Uses JPEG compression with pyramidal tiles for efficient viewing.

    Args:
        input_path: Path to input image file
        output_path: Path for output TIFF file
        quality: JPEG quality (0-100, default: 90)
        tile_size: Tile dimensions (default: 1024)

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
            Q=quality,
            tile_width=tile_size,
            tile_height=tile_size,
            strip=True,
        )
        return True
    except pyvips.Error as e:
        logger.error(f"VIPS conversion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"VIPS error: {e}")
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

        # Look for nested data.zip
        zip_files = list(extract_dir.rglob("*.zip"))
        if zip_files:
            data_path = extract_dir / "unzipped-data"
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                zf.extractall(data_path)
            return data_path

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


def is_vector_format(doc_format: str) -> bool:
    """Check if format is a vector format that converts to PMTiles.

    Args:
        doc_format: Format string (e.g., "GeoJSON", "Shapefile")

    Returns:
        True if vector format, False otherwise
    """
    from .config import VECTOR_FORMATS

    return doc_format in VECTOR_FORMATS


def is_image_format(doc_format: str) -> bool:
    """Check if format is an image format that converts to pyramidal TIFF.

    Args:
        doc_format: Format string (e.g., "JPEG", "TIFF")

    Returns:
        True if image format, False otherwise
    """
    from .config import IMAGE_FORMATS

    return doc_format in IMAGE_FORMATS
