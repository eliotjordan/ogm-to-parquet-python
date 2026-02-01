"""OpenGeoMetadata to Parquet harvester.

Converts OpenGeoMetadata JSON files into a single Parquet file
optimized for querying with DuckDB-WASM.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .geometry import Geometry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Field mapping from GeoBlacklight schema to simplified schema
FIELD_MAP = {
    "dct_title_s": "title",
    "dct_creator_sm": "creator",
    "dct_publisher_sm": "publisher",
    "dct_description_sm": "description",
    "schema_provider_s": "provider",
    "dct_accessRights_s": "access_rights",
    "gbl_resourceClass_sm": "resource_class",
    "gbl_resourceType_sm": "resource_type",
    "dcat_theme_sm": "theme",
    "dct_subject_sm": "subject",
    "dct_spatial_sm": "location",
    "dct_format_s": "format",
    "dct_identifier_sm": "identifier",
    "dct_references_s": "references",
    "dct_temporal_sm": "temporal",
    "gbl_wxsIdentifier_s": "wxs_identifier",
    "gbl_mdModified_dt": "modified",
    "locn_geometry": "geometry",
    "dcat_bbox": "bbox",
    "gbl_indexYear_im": "index_year",
}

# PyArrow schema for Parquet output
PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("creator", pa.list_(pa.string())),
        ("location", pa.list_(pa.string())),
        ("publisher", pa.list_(pa.string())),
        ("provider", pa.string()),
        ("access_rights", pa.string()),
        ("resource_class", pa.list_(pa.string())),
        ("resource_type", pa.list_(pa.string())),
        ("subject", pa.list_(pa.string())),
        ("theme", pa.list_(pa.string())),
        ("thumbnail", pa.string()),
        ("geojson", pa.string()),
        ("description", pa.string()),
        ("format", pa.string()),
        ("identifier", pa.list_(pa.string())),
        ("references", pa.string()),
        ("temporal", pa.list_(pa.string())),
        ("wxs_identifier", pa.string()),
        ("modified", pa.string()),
        ("index_year", pa.list_(pa.float64())),
        ("full_text", pa.string()),
    ]
)


class OgmToParquet:
    """Harvester that converts OpenGeoMetadata JSON files to Parquet format."""

    def __init__(
        self,
        ogm_path: str = "./tmp/opengeometadata/",
        output_path: str = "./tmp/ogm.parquet",
    ):
        """Initialize the harvester.

        Args:
            ogm_path: Path to directory containing OpenGeoMetadata JSON files
            output_path: Path for output Parquet file
        """
        self.ogm_path = Path(ogm_path)
        self.output_path = Path(output_path)
        self.rows: List[Dict[str, Any]] = []

    def convert(self) -> None:
        """Convert all JSON files to Parquet format."""
        docs = self._collect_documents()

        for doc in docs:
            try:
                # Skip non-dict documents
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping non-dict document: {type(doc)}")
                    continue

                doc_id = doc.get("id", "unknown")
                logger.info(doc_id)
                remapped_doc = self._remap_and_clean(doc)
                row = self._build_row(remapped_doc)
                self.rows.append(row)
            except Exception as e:
                doc_id = doc.get("id", "unknown") if isinstance(doc, dict) else "unknown"
                logger.warning(f"Error processing {doc_id}: {e}")
                continue

        self._write_parquet()
        logger.info(f"Successfully wrote {len(self.rows)} records to {self.output_path}")

    def _collect_documents(self) -> List[Dict[str, Any]]:
        """Recursively collect all JSON documents from ogm_path.

        Returns:
            List of parsed JSON documents
        """
        documents = []

        if not self.ogm_path.exists():
            logger.error(f"Path does not exist: {self.ogm_path}")
            return documents

        # Recursively find all .json files
        json_files = list(self.ogm_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    # Only collect dict documents (skip strings, lists, etc.)
                    if isinstance(doc, dict):
                        documents.append(doc)
                    else:
                        logger.debug(f"Skipping non-dict JSON in {json_file}: {type(doc)}")
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue

        return documents

    def _remap_and_clean(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Remap field names and clean values.

        Args:
            doc: Original document with GeoBlacklight field names

        Returns:
            Document with remapped fields and cleaned values
        """
        remapped = self._remap_doc_keys(doc)
        return self._clean_values(remapped)

    def _remap_doc_keys(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Remap document keys using FIELD_MAP.

        Args:
            doc: Original document

        Returns:
            Document with remapped keys
        """
        remapped = {}
        for key, value in doc.items():
            new_key = FIELD_MAP.get(key, key)
            remapped[new_key] = value
        return remapped

    def _clean_values(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Clean values by removing single quotes (prevents SQL injection).

        Args:
            doc: Document to clean

        Returns:
            Document with cleaned values
        """
        cleaned = {}
        for key, value in doc.items():
            if isinstance(value, list):
                cleaned[key] = self._clean_array(value)
            elif isinstance(value, str):
                cleaned[key] = value.replace("'", "")
            else:
                cleaned[key] = value
        return cleaned

    def _clean_array(self, arr: List[Any]) -> List[Any]:
        """Clean array values by removing single quotes from strings.

        Args:
            arr: Array to clean

        Returns:
            Cleaned array
        """
        cleaned = []
        for item in arr:
            if isinstance(item, str):
                cleaned.append(item.replace("'", ""))
            else:
                cleaned.append(item)
        return cleaned

    def _build_row(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Build a row for Parquet output.

        Args:
            doc: Cleaned and remapped document

        Returns:
            Dictionary with fields matching PARQUET_SCHEMA
        """
        return {
            "id": self._ensure_string(doc.get("id")),
            "title": self._ensure_string(doc.get("title")),
            "creator": self._ensure_list(doc.get("creator")),
            "location": self._ensure_list(doc.get("location")),
            "publisher": self._ensure_list(doc.get("publisher")),
            "provider": self._ensure_string(doc.get("provider")),
            "access_rights": self._ensure_string(doc.get("access_rights")),
            "resource_class": self._ensure_list(doc.get("resource_class")),
            "resource_type": self._ensure_list(doc.get("resource_type")),
            "subject": self._ensure_list(doc.get("subject")),
            "theme": self._ensure_list(doc.get("theme")),
            "thumbnail": self._extract_thumbnail_url(doc),
            "geojson": self._extract_geojson(doc),
            "description": self._ensure_string(doc.get("description")),
            "format": self._ensure_string(doc.get("format")),
            "identifier": self._ensure_list(doc.get("identifier")),
            "references": self._ensure_string(doc.get("references")),
            "temporal": self._ensure_list(doc.get("temporal")),
            "wxs_identifier": self._ensure_string(doc.get("wxs_identifier")),
            "modified": self._ensure_string(doc.get("modified")),
            "index_year": self._ensure_list(doc.get("index_year")),
            "full_text": None,  # Not populated in Ruby version either
        }

    def _ensure_list(self, value: Any) -> Optional[List[Any]]:
        """Ensure value is a list or None.

        Args:
            value: Value to convert

        Returns:
            List or None
        """
        if value is None:
            return None
        if isinstance(value, list):
            return value if value else None
        return [value]

    def _ensure_string(self, value: Any) -> Optional[str]:
        """Ensure value is a string or None.

        If value is a list, join elements with a space.
        If value is not a string, convert to string.

        Args:
            value: Value to convert

        Returns:
            String or None
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Join list elements with space, filtering out None values
            return " ".join(str(item) for item in value if item is not None) if value else None
        return str(value)

    def _extract_geojson(self, doc: Dict[str, Any]) -> Optional[str]:
        """Extract GeoJSON from bbox field.

        Args:
            doc: Document with potential bbox field

        Returns:
            GeoJSON string or None
        """
        bbox = doc.get("bbox")
        if not bbox:
            return None
        return Geometry(bbox).to_geojson()

    def _extract_thumbnail_url(self, doc: Dict[str, Any]) -> Optional[str]:
        """Extract thumbnail URL from references field.

        Prefers schema.org thumbnailUrl, falls back to IIIF image API.

        Args:
            doc: Document with potential references field

        Returns:
            Thumbnail URL or None
        """
        refs = doc.get("references")
        if not refs:
            return None

        try:
            refs_dict = json.loads(refs) if isinstance(refs, str) else refs

            # Prefer schema.org thumbnail
            if "http://schema.org/thumbnailUrl" in refs_dict:
                return refs_dict["http://schema.org/thumbnailUrl"]

            # Fall back to IIIF
            if "http://iiif.io/api/image" in refs_dict:
                iiif_url = refs_dict["http://iiif.io/api/image"]
                return iiif_url.replace("info.json", "square/150,150/0/default.jpg")

        except Exception as e:
            logger.debug(f"Error parsing references: {e}")

        return None

    def _write_parquet(self) -> None:
        """Write collected rows to Parquet file with ZSTD compression."""
        if not self.rows:
            logger.warning("No rows to write")
            return

        # Convert list of dicts to PyArrow Table
        table = pa.Table.from_pylist(self.rows, schema=PARQUET_SCHEMA)

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet with ZSTD compression
        pq.write_table(
            table,
            self.output_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )


def main():
    """Main entry point for the harvester."""
    harvester = OgmToParquet()
    harvester.convert()


if __name__ == "__main__":
    main()
