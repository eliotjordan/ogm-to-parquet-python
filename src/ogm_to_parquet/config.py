"""Centralized configuration for ogm-to-parquet.

Provides dataclass-based configuration objects for all pipeline components,
consolidating default values that were previously scattered across modules.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HarvestConfig:
    """Configuration for the OGM harvester.

    Attributes:
        ogm_path: Path to directory containing OpenGeoMetadata JSON files
        output_path: Path for output Parquet file
        model_dir: Directory to save distilled embedding model
        enable_embeddings: Whether to generate embeddings
        embedding_dims: Target dimensions for embeddings (32, 64, 128, or 256)
        max_vocab_size: Maximum vocabulary size for embeddings
    """

    ogm_path: Path = field(default_factory=lambda: Path("./tmp/opengeometadata/"))
    output_path: Path = field(default_factory=lambda: Path("./tmp/ogm.parquet"))
    model_dir: Path = field(default_factory=lambda: Path("./tmp/ogm-model/"))
    enable_embeddings: bool = True
    embedding_dims: int = 128
    max_vocab_size: int = 5000


@dataclass
class DerivativesConfig:
    """Configuration for cloud derivative processing.

    Attributes:
        db_path: Path to SQLite database with cloud_derivatives table
        scratch_dir: Directory for temporary processing files
        output_dir: Directory for output derivative files
        redis_url: Redis connection URL for RQ job queue
        tippecanoe_timeout: Timeout in seconds for tippecanoe processing
        max_retries: Maximum retry attempts for failed jobs
        delay: Delay in seconds between job submissions (rate limiting)
        workers: Number of worker processes to spawn
    """

    db_path: Path = field(default_factory=lambda: Path("./tmp/derivatives.db"))
    scratch_dir: Path = field(default_factory=lambda: Path("./tmp/scratch"))
    output_dir: Path = field(default_factory=lambda: Path("./tmp/cloud_derivatives"))
    redis_url: str = "redis://localhost:6379"
    tippecanoe_timeout: int = 1800  # 30 minutes
    max_retries: int = 5
    delay: float = 1.0
    workers: int = 5


@dataclass
class TextExtractionConfig:
    """Configuration for text extraction from map images.

    Attributes:
        db_path: Path to SQLite database with text_extraction table
        scratch_dir: Directory for temporary file downloads
        ollama_url: Base URL for Ollama API
        model: Ollama model to use for vision tasks
        max_retries: Maximum retries for API calls
        timeout: Request timeout in seconds
    """

    db_path: Path = field(default_factory=lambda: Path("./tmp/text_extraction.db"))
    scratch_dir: Path = field(default_factory=lambda: Path("./tmp/scratch"))
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3-vl:30b"
    max_retries: int = 3
    timeout: int = 300


@dataclass
class EnrichmentConfig:
    """Configuration for enrichment preparation.

    Attributes:
        derivatives_db_path: Path to derivatives SQLite database
        text_extraction_db_path: Path to text extraction SQLite database
    """

    derivatives_db_path: Path = field(default_factory=lambda: Path("./tmp/derivatives.db"))
    text_extraction_db_path: Path = field(default_factory=lambda: Path("./tmp/text_extraction.db"))


# Queue name for RQ derivative processing
QUEUE_NAME = "derivatives"

# Reference keys for document filtering
DOWNLOAD_URL_KEY = "http://schema.org/downloadUrl"
IIIF_IMAGE_KEY = "http://iiif.io/api/image"

# Reference keys that indicate tile/map services (exclude from cloud_derivatives)
EXCLUDED_SERVICE_KEYS = frozenset(
    {
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
        "http://iiif.io/api/presentation#manifest",
        "http://iiif.io/api/image",
    }
)

# Formats eligible for cloud derivative generation
DERIVATIVE_FORMATS = frozenset(
    {"GeoJSON", "GeoPackage", "KML", "Shapefile", "JPEG", "JPEG2000", "TIFF"}
)

# Vector formats that convert to PMTiles
VECTOR_FORMATS = frozenset({"GeoJSON", "GeoPackage", "KML", "Shapefile"})

# Image formats that convert to pyramidal TIFF
IMAGE_FORMATS = frozenset({"JPEG", "JPEG2000", "TIFF"})

# File extensions for vector formats
VECTOR_EXTENSIONS = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".gpkg": "GeoPackage",
    ".kml": "KML",
    ".shp": "Shapefile",
}

# File extensions for image formats
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".jp2", ".tif", ".tiff"})

# Valid image extensions for text extraction
VALID_IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
        ".webp",
        ".jp2",
    }
)

# Maximum image size for Ollama (9MB)
MAX_IMAGE_SIZE_BYTES = 9 * 1024 * 1024
