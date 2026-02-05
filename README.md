# OpenGeoMetadata to Parquet Converter (Python)

Python implementation of the OpenGeoMetadata to Parquet converter with enrichment pipelines. This tool:

1. **Harvests** OpenGeoMetadata JSON files into a Parquet file with semantic embeddings
2. **Generates cloud derivatives** - converts vector data to PMTiles and images to pyramidal TIFFs
3. **Extracts text** from map images using vision AI models

## Requirements

- Python 3.11+
- uv (for package management)
- Docker (for Redis)
- GDAL (ogr2ogr) and tippecanoe (for derivatives)
- Ollama (optional, for text extraction)

## Installation

```bash
# Install dependencies
uv sync --all-extras

# Start Redis (required for derivatives processing)
docker compose up -d
```

## Quick Start

```bash
# 1. Download and harvest OGM metadata
uv run ogm-harvest --download

# 2. Prepare enrichment tasks
uv run ogm-enrich-prepare

# 3. Process derivatives (vector → PMTiles, images → pyramidal TIFFs)
uv run ogm-enrich-derivatives

# 4. Extract text from map images (requires Ollama)
uv run ogm-enrich-extract
```

## Commands

### ogm-harvest - Convert OGM to Parquet

Converts OpenGeoMetadata JSON files to a queryable Parquet file with semantic embeddings.

```bash
# Download OGM repositories and harvest
uv run ogm-harvest --download

# Harvest only (from existing tmp/opengeometadata/)
uv run ogm-harvest

# Custom embedding model size
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000

# Without embeddings
uv run ogm-harvest --no-embeddings
```

**Outputs:**
- `tmp/ogm.parquet` - Harvested metadata with embeddings
- `tmp/ogm-model/` - Distilled embedding model for browser use

### ogm-enrich-prepare - Prepare Enrichment Tasks

Scans harvested documents and creates enrichment tasks in SQLite databases.

```bash
uv run ogm-enrich-prepare
```

**Outputs:**
- `tmp/derivatives.db` - `cloud_derivatives` table for vector/image files to convert
- `tmp/text_extraction.db` - `text_extraction` table for images for OCR

### ogm-enrich-derivatives - Process Cloud Derivatives

Converts vector files to PMTiles and images to pyramidal TIFFs using a Redis-backed job queue.

```bash
# Full processing (enqueue + start workers)
uv run ogm-enrich-derivatives

# Start workers for existing queue (after restart)
uv run ogm-enrich-derivatives --workers-only

# Preview without processing
uv run ogm-enrich-derivatives --dry-run

# Just enqueue, run workers separately
uv run ogm-enrich-derivatives --enqueue-only

# Check queue statistics
uv run ogm-enrich-derivatives --stats

# Process single document
uv run ogm-enrich-derivatives --id "stanford-abc123"

# Custom settings
uv run ogm-enrich-derivatives \
  --workers 4 \
  --delay 2.0 \
  --tippecanoe-timeout 3600 \
  --max-retries 5

# Monitor with rq-dashboard (web UI at http://localhost:9181)
rq-dashboard --redis-url redis://localhost:6379
```

**Processing:**
- **Vector formats** (GeoJSON, Shapefile, GeoPackage, KML):
  - ogr2ogr → FlatGeobuf (reprojected to EPSG:4326)
  - tippecanoe → PMTiles
- **Image formats** (JPEG, JPEG2000, TIFF):
  - pyvips → Pyramidal TIFF (JPEG compression, 1024x1024 tiles, Q=90)

**Outputs:**
```
tmp/cloud_derivatives/
  ab/c1/23/abc123def/dataset.pmtiles   # Vector
  pn/86/3f/pn863fv0810/dataset.tif     # Image
```

### ogm-enrich-extract - Extract Text from Maps

Uses Ollama vision models to extract text from map images.

```bash
# Process all pending images
uv run ogm-enrich-extract

# Process single document
uv run ogm-enrich-extract --id "doc123"

# Custom model
uv run ogm-enrich-extract --model qwen2.5-vl:32b
```

**Requires:** Ollama running with a vision model (e.g., `qwen2.5-vl:32b`)

## Development

```bash
# Run tests with coverage (recommended - avoids segfault on macOS)
./scripts/run_tests.sh

# Run tests (may segfault on macOS due to C library conflicts)
uv run pytest

# Run specific test file
uv run pytest tests/test_derivatives.py

# Lint code
uv run ruff check src tests

# Format code
uv run ruff format src tests
```

## Project Structure

```
src/ogm_to_parquet/
├── harvest.py          # OGM JSON → Parquet conversion
├── embeddings.py       # Semantic embedding generation
├── geometry.py         # WKT/ENVELOPE → GeoJSON conversion
├── download.py         # OGM repository download
├── enrichment.py       # Enrichment task preparation
├── derivatives.py      # Derivative processing orchestration
├── derivative_jobs.py  # RQ job functions (ogr2ogr, tippecanoe, pyvips)
└── text_extract.py     # Vision AI text extraction

tests/
├── test_harvest.py
├── test_geometry.py
├── test_embeddings.py
├── test_enrichment.py
├── test_derivatives.py
└── test_text_extract.py
```

## Database Schema

Enrichment tasks are tracked in two separate databases:

**tmp/derivatives.db:**
```sql
CREATE TABLE cloud_derivatives (
    id TEXT PRIMARY KEY,
    download_url TEXT,
    format TEXT,           -- GeoJSON, Shapefile, JPEG, etc.
    derivative_url TEXT,   -- Output path when complete
    status TEXT,           -- unprocessed, enqueued, in_progress, complete, error
    error_message TEXT,
    retry_count INTEGER
);
```

**tmp/text_extraction.db:**
```sql
CREATE TABLE text_extraction (
    id TEXT PRIMARY KEY,
    image_url TEXT,
    format TEXT,           -- IIIF, JPEG, TIFF
    generated_output TEXT, -- JSON with extracted text
    status TEXT,           -- unprocessed, in_progress, complete, error
    error_message TEXT
);
```

**Status flow:** `unprocessed` → `enqueued` → `in_progress` → `complete`/`error`

## Common Operations

### Check Processing Status

```bash
# Derivatives status
sqlite3 tmp/derivatives.db "SELECT status, COUNT(*) FROM cloud_derivatives GROUP BY status"

# Text extraction status
sqlite3 tmp/text_extraction.db "SELECT status, COUNT(*) FROM text_extraction GROUP BY status"
```

### Reset Failed Jobs

```bash
# Reset failed derivatives
sqlite3 tmp/derivatives.db "UPDATE cloud_derivatives SET status='unprocessed' WHERE status='error'"

# Reset all derivatives
sqlite3 tmp/derivatives.db "UPDATE cloud_derivatives SET status='unprocessed'"

# Reset failed text extraction
sqlite3 tmp/text_extraction.db "UPDATE text_extraction SET status='unprocessed' WHERE status='error'"
```

### Monitor Redis Queue

```bash
# Queue statistics (snapshot)
uv run ogm-enrich-derivatives --stats

# Real-time monitoring with rq-dashboard (web UI at http://localhost:9181)
rq-dashboard --redis-url redis://localhost:6379
```

## Using with DuckDB

Query the harvested Parquet file:

```sql
-- Search by location
SELECT * FROM 'tmp/ogm.parquet'
WHERE ST_Intersects(
  ST_GeomFromGeoJSON(geojson),
  ST_MakeEnvelope(-122.5, 37.7, -122.0, 37.8)
);

-- Semantic search (using embeddings)
SELECT title,
       list_cosine_similarity(embeddings, $query_embedding) as score
FROM 'tmp/ogm.parquet'
ORDER BY score DESC
LIMIT 10;
```

## Browser Integration

The distilled embedding model (`tmp/ogm-model/`) can be loaded in JavaScript for client-side semantic search:

```javascript
import { Tokenizer } from "@huggingface/tokenizers";

// Load model
const tokenizer = await Tokenizer.from_pretrained("./ogm-model/tokenizer.json");
const response = await fetch("./ogm-model/embeddings.bin");
const embeddingMatrix = new Float32Array(await response.arrayBuffer());

// Encode query
function encodeQuery(text) {
  const tokens = tokenizer.encode(text).ids;
  const embedding = new Float32Array(256);
  for (const id of tokens) {
    for (let i = 0; i < 256; i++) {
      embedding[i] += embeddingMatrix[id * 256 + i];
    }
  }
  return embedding.map(v => v / tokens.length);
}
```

## Differences from Ruby Version

This Python implementation adds:

1. **Semantic embeddings** with Model2Vec
2. **Cloud derivatives pipeline** (PMTiles, pyramidal TIFFs)
3. **Text extraction** from map images
4. Comprehensive test suite (100+ tests)
5. Type hints throughout
6. Modern tooling (uv, ruff, pytest)
