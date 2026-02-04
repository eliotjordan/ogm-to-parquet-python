"""Enrichment preparation for OpenGeoMetadata documents.

Creates and populates SQLite tables for tracking cloud derivative generation
and text extraction tasks.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Reference keys that indicate tile/map services (exclude from cloud_derivatives)
EXCLUDED_SERVICE_KEYS = {
    "urn:x-esri:serviceType:ArcGIS#DynamicMapLayer",
    "urn:x-esri:serviceType:ArcGIS#FeatureLayer",
    "urn:x-esri:serviceType:ArcGIS#ImageMapLayer",
    "urn:x-esri:serviceType:ArcGIS#TiledMapLayer",
    "https://github.com/protomaps/PMTiles",
    "https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification",
    "https://github.com/mapbox/tilejson-spec",
    "http://www.opengis.net/def/serviceType/ogc/wms",
    "http://www.opengis.net/def/serviceType/ogc/wmts",
    "https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames",
}

# Formats eligible for cloud derivative generation
DERIVATIVE_FORMATS = {"GeoJSON", "GeoPackage", "KML", "Shapefile", "JPEG", "JPEG2000", "TIFF"}

# Formats eligible for text extraction (image formats)
IMAGE_FORMATS = {"JPEG", "JPEG2000", "TIFF"}

# Reference keys
DOWNLOAD_URL_KEY = "http://schema.org/downloadUrl"
IIIF_IMAGE_KEY = "http://iiif.io/api/image"


class EnrichmentPreparer:
    """Prepares enrichment tasks by populating SQLite tracking tables."""

    def __init__(
        self,
        db_path: str = "./tmp/enrichment.db",
    ):
        """Initialize the enrichment preparer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)

    def setup_database(self) -> bool:
        """Create the SQLite database and tables if they don't exist.

        Returns:
            True if database was created, False if it already existed
        """
        db_existed = self.db_path.exists()

        if db_existed:
            logger.info(f"Database already exists: {self.db_path}")
            return False

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Create cloud_derivatives table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloud_derivatives (
                    id TEXT PRIMARY KEY,
                    download_url TEXT,
                    format TEXT,
                    derivative_url TEXT,
                    status TEXT,
                    error_message TEXT
                )
            """)

            # Create text_extraction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_extraction (
                    id TEXT PRIMARY KEY,
                    image_url TEXT,
                    format TEXT,
                    generated_output TEXT,
                    status TEXT,
                    error_message TEXT
                )
            """)

            conn.commit()
            logger.info(f"Created database with tables: {self.db_path}")
            return True
        finally:
            conn.close()

    def prepare(self, documents: list[dict[str, Any]]) -> dict[str, int]:
        """Prepare enrichment tasks from documents.

        Populates the cloud_derivatives and text_extraction tables based on
        document filtering criteria.

        Args:
            documents: List of Aardvark JSON documents

        Returns:
            Dictionary with counts of inserted records per table
        """
        # Ensure database exists
        self.setup_database()

        conn = sqlite3.connect(self.db_path)
        try:
            cloud_derivatives_count = self._populate_cloud_derivatives(conn, documents)
            text_extraction_count = self._populate_text_extraction(conn, documents)

            return {
                "cloud_derivatives": cloud_derivatives_count,
                "text_extraction": text_extraction_count,
            }
        finally:
            conn.close()

    def _populate_cloud_derivatives(
        self, conn: sqlite3.Connection, documents: list[dict[str, Any]]
    ) -> int:
        """Populate cloud_derivatives table with eligible documents.

        Filtering criteria:
        - dct_accessRights_s must be "open"
        - dct_format_s must be one of the DERIVATIVE_FORMATS
        - dct_references_s must NOT contain any EXCLUDED_SERVICE_KEYS
        - dct_references_s must contain DOWNLOAD_URL_KEY

        Args:
            conn: SQLite connection
            documents: List of documents to filter

        Returns:
            Number of records inserted
        """
        cursor = conn.cursor()
        inserted = 0
        skipped_existing = 0

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            # Check access rights
            if doc.get("dct_accessRights_s") != "Public":
                continue

            # Check format
            doc_format = doc.get("dct_format_s")
            if doc_format not in DERIVATIVE_FORMATS:
                continue

            # Parse and check references
            references = self._parse_references(doc.get("dct_references_s"))
            if references is None:
                continue

            # Must not contain excluded service keys
            if any(key in references for key in EXCLUDED_SERVICE_KEYS):
                continue

            # Must contain download URL
            if DOWNLOAD_URL_KEY not in references:
                continue

            # Check if already exists
            cursor.execute("SELECT 1 FROM cloud_derivatives WHERE id = ?", (doc_id,))
            if cursor.fetchone():
                skipped_existing += 1
                continue

            # Extract download URL
            download_url = self._extract_download_url(references[DOWNLOAD_URL_KEY]) or ""

            # Insert new record
            cursor.execute(
                """
                INSERT INTO cloud_derivatives (id, download_url, format, derivative_url, status)
                VALUES (?, ?, ?, ?, ?)
            """,
                (doc_id, download_url, doc_format or "", "", "unprocessed"),
            )
            inserted += 1

        conn.commit()

        if skipped_existing > 0:
            logger.info(f"cloud_derivatives: skipped {skipped_existing} existing records")
        logger.info(f"cloud_derivatives: inserted {inserted} new records")

        return inserted

    def _populate_text_extraction(
        self, conn: sqlite3.Connection, documents: list[dict[str, Any]]
    ) -> int:
        """Populate text_extraction table with eligible documents.

        Filtering criteria:
        - dct_accessRights_s must be "open"
        - Either:
          - dct_references_s contains IIIF_IMAGE_KEY, OR
          - dct_format_s is an IMAGE_FORMAT AND dct_references_s contains DOWNLOAD_URL_KEY

        Args:
            conn: SQLite connection
            documents: List of documents to filter

        Returns:
            Number of records inserted
        """
        cursor = conn.cursor()
        inserted = 0
        skipped_existing = 0

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            # Check access rights
            if doc.get("dct_accessRights_s") != "Public":
                continue

            # Parse references
            references = self._parse_references(doc.get("dct_references_s"))
            if references is None:
                continue

            # Check eligibility: IIIF OR (image format AND download URL)
            has_iiif = IIIF_IMAGE_KEY in references
            doc_format = doc.get("dct_format_s")
            has_image_download = doc_format in IMAGE_FORMATS and DOWNLOAD_URL_KEY in references

            if not (has_iiif or has_image_download):
                continue

            # Check if already exists
            cursor.execute("SELECT 1 FROM text_extraction WHERE id = ?", (doc_id,))
            if cursor.fetchone():
                skipped_existing += 1
                continue

            # Generate image URL
            image_url = self._generate_image_url(references) or ""

            # Determine format: "IIIF" if IIIF reference exists, otherwise use dct_format_s
            format_value = "IIIF" if has_iiif else (doc_format or "")

            # Insert new record
            cursor.execute(
                """
                INSERT INTO text_extraction (id, image_url, format, generated_output, status)
                VALUES (?, ?, ?, ?, ?)
            """,
                (doc_id, image_url, format_value, "", "unprocessed"),
            )
            inserted += 1

        conn.commit()

        if skipped_existing > 0:
            logger.info(f"text_extraction: skipped {skipped_existing} existing records")
        logger.info(f"text_extraction: inserted {inserted} new records")

        return inserted

    def _parse_references(self, references: Any) -> dict[str, str] | None:
        """Parse the dct_references_s field as JSON.

        Args:
            references: Raw references value (string or dict)

        Returns:
            Parsed dictionary or None if invalid
        """
        if not references:
            return None

        if isinstance(references, dict):
            return references

        if isinstance(references, str):
            try:
                return json.loads(references)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_download_url(self, download_value: Any) -> str | None:
        """Extract download URL from the downloadUrl reference value.

        The downloadUrl can be in two formats:
        1. A string: "https://example.com/file.zip"
        2. An array of objects: [{"url": "https://...", "label": "Shapefile"}, ...]

        Args:
            download_value: The value from the downloadUrl reference key

        Returns:
            The download URL string, or None if not found
        """
        if isinstance(download_value, str):
            return download_value

        if isinstance(download_value, list) and download_value:
            # Take the first download object's URL
            first_download = download_value[0]
            if isinstance(first_download, dict) and "url" in first_download:
                return first_download["url"]

        return None

    def _generate_image_url(self, references: dict[str, Any]) -> str | None:
        """Generate an image URL from document references.

        Priority:
        1. IIIF image URL (transformed to full image request)
        2. Download URL (used directly)

        Args:
            references: Parsed references dictionary

        Returns:
            Image URL string or None if no suitable reference found
        """
        # Prefer IIIF image reference
        if IIIF_IMAGE_KEY in references:
            iiif_url = references[IIIF_IMAGE_KEY]
            # Replace info.json with full image parameters
            if iiif_url.endswith("info.json"):
                # Remove "info.json" (9 chars) and any trailing slash
                base_url = iiif_url[:-9].rstrip("/")
                return base_url + "/full/full/0/default.jpg"
            # If no info.json suffix, strip trailing slash and append IIIF parameters
            return iiif_url.rstrip("/") + "/full/full/0/default.jpg"

        # Fall back to download URL
        if DOWNLOAD_URL_KEY in references:
            return self._extract_download_url(references[DOWNLOAD_URL_KEY])

        return None


def main():
    """Command-line entry point for enrichment preparation."""
    import argparse

    from .harvest import OgmToParquet

    parser = argparse.ArgumentParser(
        description="Prepare enrichment tasks from OpenGeoMetadata documents"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./tmp/enrichment.db",
        help="Path to SQLite database (default: ./tmp/enrichment.db)",
    )
    parser.add_argument(
        "--ogm-path",
        type=str,
        default="./tmp/opengeometadata/",
        help="Path to OpenGeoMetadata data directory (default: ./tmp/opengeometadata/)",
    )

    args = parser.parse_args()

    # Collect documents using harvester's collection logic
    harvester = OgmToParquet(ogm_path=args.ogm_path)
    documents = harvester._collect_documents()
    logger.info(f"Collected {len(documents)} valid Aardvark documents")

    # Prepare enrichment
    preparer = EnrichmentPreparer(db_path=args.db_path)
    results = preparer.prepare(documents)

    logger.info(f"Enrichment preparation complete: {results}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
