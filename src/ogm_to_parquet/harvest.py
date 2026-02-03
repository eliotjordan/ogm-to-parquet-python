"""OpenGeoMetadata to Parquet harvester.

Converts OpenGeoMetadata JSON files into a single Parquet file
optimized for querying with DuckDB-WASM.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .download import download_repositories
from .embeddings import EmbeddingGenerator, VocabularyBuilder, distill_model
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
        ("embeddings", pa.list_(pa.float32())),
    ]
)


class OgmToParquet:
    """Harvester that converts OpenGeoMetadata JSON files to Parquet format."""

    def __init__(
        self,
        ogm_path: str = "./tmp/opengeometadata/",
        output_path: str = "./tmp/ogm.parquet",
        model_dir: str = "./tmp/ogm-model/",
        enable_embeddings: bool = True,
        embedding_dims: int = 128,
        max_vocab_size: int = 10000,
    ):
        """Initialize the harvester.

        Args:
            ogm_path: Path to directory containing OpenGeoMetadata JSON files
            output_path: Path for output Parquet file
            model_dir: Directory to save distilled embedding model
            enable_embeddings: Whether to generate embeddings (requires model2vec)
            embedding_dims: Target dimensions for embeddings (lower = smaller model)
                           Recommended: 256 (default), 128 (smaller), 64 (smallest)
            max_vocab_size: Maximum vocabulary size (lower = smaller model)
                           Recommended: 5000 (default, ~15MB), 2000 (~8MB), 1000 (~5MB)
        """
        self.ogm_path = Path(ogm_path)
        self.output_path = Path(output_path)
        self.model_dir = Path(model_dir)
        self.enable_embeddings = enable_embeddings
        self.embedding_dims = embedding_dims
        self.max_vocab_size = max_vocab_size
        self.rows: list[dict[str, Any]] = []
        self.embedding_generator: EmbeddingGenerator | None = None

    def convert(self) -> None:
        """Convert all JSON files to Parquet format with embeddings."""
        docs = self._collect_documents()

        # Build vocabulary and distill model before processing documents
        if self.enable_embeddings:
            try:
                self._prepare_embedding_model(docs)
            except Exception as e:
                logger.error(f"Failed to prepare embedding model: {e}")
                logger.warning("Continuing without embeddings")
                self.enable_embeddings = False

        # Process each document
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

    def _prepare_embedding_model(self, docs: list[dict[str, Any]]) -> None:
        """Build vocabulary, distill model, and create embedding generator.

        Args:
            docs: All documents to process (needed for vocabulary building)
        """
        logger.info("Preparing embedding model...")

        # Remap all documents for vocabulary building
        remapped_docs = []
        for doc in docs:
            if isinstance(doc, dict):
                remapped_docs.append(self._remap_and_clean(doc))

        # Build vocabulary from all documents
        # Reduce vocabulary for smaller models
        vocab_builder = VocabularyBuilder(
            max_vocab_size=self.max_vocab_size,
            min_term_freq=2,
            include_bigrams=self.embedding_dims >= 128,  # Skip bigrams for very small models
            limit_controlled_vocab=self.max_vocab_size
            < 10000,  # Limit controlled vocab for small models
        )
        vocabulary = vocab_builder.build_vocabulary(remapped_docs)

        # Distill model with custom vocabulary
        model_path = distill_model(
            vocabulary=vocabulary,
            output_dir=str(self.model_dir),
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            pca_dims=self.embedding_dims,
        )

        # Create embedding generator
        self.embedding_generator = EmbeddingGenerator(str(model_path))
        logger.info("Embedding model ready")

    def _collect_documents(self) -> list[dict[str, Any]]:
        """Recursively collect all JSON documents from ogm_path.

        Only includes documents that have an 'id' field and 'gbl_mdVersion_s'
        set to 'Aardvark' (the current GeoBlacklight metadata schema version).

        Returns:
            List of parsed JSON documents
        """
        documents = []
        skipped_no_id = 0
        skipped_not_aardvark = 0

        if not self.ogm_path.exists():
            logger.error(f"Path does not exist: {self.ogm_path}")
            return documents

        # Recursively find all .json files
        json_files = list(self.ogm_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    doc = json.load(f)

                    # Only collect dict documents (skip strings, lists, etc.)
                    if not isinstance(doc, dict):
                        logger.debug(f"Skipping non-dict JSON in {json_file}: {type(doc)}")
                        continue

                    # Require id field
                    if not doc.get("id"):
                        skipped_no_id += 1
                        logger.debug(f"Skipping document without id: {json_file}")
                        continue

                    # Require Aardvark schema version
                    if doc.get("gbl_mdVersion_s") != "Aardvark":
                        skipped_not_aardvark += 1
                        logger.debug(f"Skipping non-Aardvark document: {json_file}")
                        continue

                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue

        if skipped_no_id > 0:
            logger.info(f"Skipped {skipped_no_id} documents without id")
        if skipped_not_aardvark > 0:
            logger.info(f"Skipped {skipped_not_aardvark} non-Aardvark documents")

        return documents

    def _remap_and_clean(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Remap field names and clean values.

        Args:
            doc: Original document with GeoBlacklight field names

        Returns:
            Document with remapped fields and cleaned values
        """
        remapped = self._remap_doc_keys(doc)
        return self._clean_values(remapped)

    def _remap_doc_keys(self, doc: dict[str, Any]) -> dict[str, Any]:
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

    def _clean_values(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Clean values by removing single quotes (prevents SQL injection).

        Preserves numeric types (int, float) to maintain schema compatibility.

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
                # Preserve numeric and other non-string types
                cleaned[key] = value
        return cleaned

    def _clean_array(self, arr: list[Any]) -> list[Any]:
        """Clean array values by removing single quotes from strings.

        Preserves numeric types (int, float) in arrays.

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
                # Preserve numeric and other non-string types
                cleaned.append(item)
        return cleaned

    def _build_row(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Build a row for Parquet output.

        Args:
            doc: Cleaned and remapped document

        Returns:
            Dictionary with fields matching PARQUET_SCHEMA
        """
        geojson = self._extract_geojson(doc)

        # Generate embedding if available
        embeddings = None
        if self.embedding_generator is not None:
            embeddings = self.embedding_generator.generate_embedding(doc)
            if embeddings is None:
                doc_id = doc.get("id", "unknown")
                logger.warning(f"Failed to generate embedding for document {doc_id}")

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
            "geojson": geojson,
            "description": self._ensure_string(doc.get("description")),
            "format": self._ensure_string(doc.get("format")),
            "identifier": self._ensure_list(doc.get("identifier")),
            "references": self._ensure_string(doc.get("references")),
            "temporal": self._ensure_list(doc.get("temporal")),
            "wxs_identifier": self._ensure_string(doc.get("wxs_identifier")),
            "modified": self._ensure_string(doc.get("modified")),
            "index_year": self._ensure_float_list(doc.get("index_year")),
            "embeddings": embeddings,
        }

    def _ensure_list(self, value: Any) -> list[Any] | None:
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

    def _ensure_float_list(self, value: Any) -> list[float] | None:
        """Ensure value is a list of floats or None.

        Converts strings to floats where possible.

        Args:
            value: Value to convert

        Returns:
            List of floats or None
        """
        if value is None:
            return None

        # Convert to list first
        if not isinstance(value, list):
            value = [value]

        if not value:  # Empty list
            return None

        # Convert all items to floats
        float_list = []
        for item in value:
            if item is None:
                continue
            try:
                if isinstance(item, (int, float)):
                    float_list.append(float(item))
                elif isinstance(item, str):
                    # Try to convert string to float
                    float_list.append(float(item))
            except (ValueError, TypeError):
                # Skip items that can't be converted
                logger.debug(f"Could not convert {item} to float")
                continue

        return float_list if float_list else None

    def _ensure_string(self, value: Any) -> str | None:
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

    def _extract_geojson(self, doc: dict[str, Any]) -> str | None:
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

    def _extract_thumbnail_url(self, doc: dict[str, Any]) -> str | None:
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert OpenGeoMetadata JSON to Parquet with embeddings"
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=128,
        choices=[32, 64, 128, 256],
        help="Embedding dimensions (lower = smaller model). Default: 128",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=5000,
        help="Maximum vocabulary size (lower = smaller model). Default: 10000",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding generation",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/update OpenGeoMetadata repositories before harvesting",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/update repositories without harvesting (implies --download)",
    )
    parser.add_argument(
        "--repos-config",
        type=str,
        default=None,
        help="Path to YAML file with list of repositories to download (default: built-in repos.yaml)",
    )
    parser.add_argument(
        "--ogm-path",
        type=str,
        default="./tmp/opengeometadata/",
        help="Path to OpenGeoMetadata data directory (default: ./tmp/opengeometadata/)",
    )

    args = parser.parse_args()

    # Download repositories if requested (--download-only implies --download)
    if args.download or args.download_only:
        logger.info("Downloading OpenGeoMetadata repositories...")
        results = download_repositories(
            target_dir=args.ogm_path,
            config_path=args.repos_config,
        )
        failed = [name for name, success in results.items() if not success]
        if failed:
            logger.warning(f"Failed to download/update: {', '.join(failed)}")

        # Exit early if download-only mode
        if args.download_only:
            successful = sum(1 for v in results.values() if v)
            logger.info(f"Download complete: {successful}/{len(results)} repositories")
            return

    harvester = OgmToParquet(
        ogm_path=args.ogm_path,
        enable_embeddings=not args.no_embeddings,
        embedding_dims=args.embedding_dims,
        max_vocab_size=args.max_vocab_size,
    )
    harvester.convert()


if __name__ == "__main__":
    main()
