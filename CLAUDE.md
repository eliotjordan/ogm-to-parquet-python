# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Base instructions

You are a senior software engineer with 15 years of experience.
Your goal is to write production-quality, maintainable, and readable code.
When I give you a task:
1. Write clean, idiomatic code in the requested language.
2. Follow established design patterns where relevant.
3. Add concise comments for non-obvious parts.
4. Use descriptive variable and function names.
5. Make the code easy to extend and test.
6. Suggest at least one improvement for scalability or clarity after the code.
7. Please tone down the overly positive tone and don't use emojis in conversation, text, or documents.

## Overview

Python tool that converts OpenGeoMetadata JSON files into a single Parquet file optimized for querying with DuckDB, plus enrichment pipelines for generating cloud-optimized derivatives (PMTiles, pyramidal TIFFs) and extracting text from map images.

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Start Redis (required for derivatives processing)
docker compose up -d

# Run harvester with default settings
uv run ogm-harvest

# Prepare enrichment tasks
uv run ogm-enrich-prepare

# Process derivatives (vector → PMTiles, image → pyramidal TIFF)
uv run ogm-enrich-derivatives

# Extract text from map images
uv run ogm-enrich-extract

# Run tests
uv run pytest

# Lint code
uv run ruff check src tests
```

## CLI Commands

### ogm-harvest - Convert OGM to Parquet

```bash
# Default settings (128 dims, 5K vocab)
uv run ogm-harvest

# Download OGM repositories first
uv run ogm-harvest --download

# Custom model size
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000

# Without embeddings
uv run ogm-harvest --no-embeddings
```

### ogm-enrich-prepare - Prepare Enrichment Tasks

Scans harvested documents and populates the enrichment database with tasks:

```bash
uv run ogm-enrich-prepare
```

Creates `tmp/enrichment.db` with two tables:
- `cloud_derivatives` - Vector/image files to convert
- `text_extraction` - Images for OCR text extraction

### ogm-enrich-derivatives - Process Cloud Derivatives

Converts vector files to PMTiles and images to pyramidal TIFFs using RQ job queue:

```bash
# Full processing (enqueue + workers + monitor)
uv run ogm-enrich-derivatives

# Start workers for existing queue (after restart)
uv run ogm-enrich-derivatives --workers-only

# Preview what would be processed
uv run ogm-enrich-derivatives --dry-run

# Just enqueue jobs (run workers separately)
uv run ogm-enrich-derivatives --enqueue-only

# Monitor existing queue
uv run ogm-enrich-derivatives --monitor-only

# Print queue statistics
uv run ogm-enrich-derivatives --stats

# Process single document
uv run ogm-enrich-derivatives --id "stanford-abc123"

# Custom settings
uv run ogm-enrich-derivatives --workers 4 --delay 2.0 --tippecanoe-timeout 3600
```

**Requires Redis** - Start with `docker compose up -d`

### ogm-enrich-extract - Extract Text from Maps

Uses Ollama vision models to extract text from map images:

```bash
uv run ogm-enrich-extract

# Process single document
uv run ogm-enrich-extract --id "doc123"

# Custom Ollama settings
uv run ogm-enrich-extract --ollama-url http://localhost:11434 --model qwen2.5-vl:32b
```

## Architecture

### Core Components

**src/ogm_to_parquet/harvest.py** - Main harvester:
- Converts OGM JSON files to Parquet with embeddings
- Field mapping, data cleaning, geometry conversion
- Outputs `tmp/ogm.parquet`

**src/ogm_to_parquet/enrichment.py** - Enrichment preparation:
- Filters documents for cloud derivatives and text extraction
- Populates SQLite database with tasks
- Outputs `tmp/enrichment.db`

**src/ogm_to_parquet/derivatives.py** - Derivative processing orchestration:
- `DerivativeProcessor` class manages RQ job queue
- Starts workers, monitors progress, handles graceful shutdown
- Uses Redis for job queue (requires `docker compose up -d`)

**src/ogm_to_parquet/derivative_jobs.py** - RQ job functions:
- `process_derivative()` - Main job function for workers
- Vector files: ogr2ogr → FlatGeobuf → tippecanoe → PMTiles
- Image files: pyvips → pyramidal TIFF (JPEG compression, 1024x1024 tiles)
- Handles retries, status updates, cleanup

**src/ogm_to_parquet/text_extract.py** - Text extraction:
- Downloads map images, processes with Ollama vision model
- Extracts categorized text (titles, legends, place names, etc.)

### Database Schema (enrichment.db)

**cloud_derivatives table:**
```sql
CREATE TABLE cloud_derivatives (
    id TEXT PRIMARY KEY,
    download_url TEXT,
    format TEXT,              -- GeoJSON, Shapefile, JPEG, TIFF, etc.
    derivative_url TEXT,      -- Output file path when complete
    status TEXT,              -- unprocessed, enqueued, in_progress, complete, error
    error_message TEXT,
    retry_count INTEGER
);
```

**Status flow:** `unprocessed` → `enqueued` → `in_progress` → `complete`/`error`

### Output Directory Structure

Derivatives are stored with nested paths based on cleaned document IDs:

```
tmp/cloud_derivatives/
  pn/86/3f/pn863fv0810/dataset.pmtiles   # Vector → PMTiles
  ab/c1/23/abc123def/dataset.tif          # Image → pyramidal TIFF
```

ID prefixes (`ark-NNNNN-`, `rutgers-lib:`, `stanford-`) are stripped for cleaner paths.

## Dependencies

**Core:**
- `pyarrow` - Parquet file writing
- `shapely` - Geometry transformations
- `model2vec` - Embedding generation
- `rq`, `redis` - Job queue for derivatives
- `pyvips` - Image processing (pyramidal TIFFs)
- `requests`, `pillow` - File handling

**External tools required:**
- `ogr2ogr` (GDAL) - Vector format conversion
- `tippecanoe` - PMTiles generation
- Redis server - Job queue backend
- Ollama - Text extraction (optional)

## Testing

```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/test_derivatives.py

# With coverage
uv run pytest --cov

# Verbose output
uv run pytest -v
```

**Test files:**
- `test_harvest.py` - Harvester tests
- `test_geometry.py` - Geometry conversion tests
- `test_embeddings.py` - Embedding generation tests
- `test_enrichment.py` - Enrichment preparation tests
- `test_text_extract.py` - Text extraction tests
- `test_derivatives.py` - Derivatives processing tests

## Common Operations

### Reset and Reprocess Derivatives

```bash
# Check current status
sqlite3 tmp/enrichment.db "SELECT status, COUNT(*) FROM cloud_derivatives GROUP BY status"

# Reset failed jobs to unprocessed
sqlite3 tmp/enrichment.db "UPDATE cloud_derivatives SET status='unprocessed' WHERE status='error'"

# Reset all to reprocess
sqlite3 tmp/enrichment.db "UPDATE cloud_derivatives SET status='unprocessed'"
```

### Monitor Redis Queue

```bash
# Check queue stats
uv run ogm-enrich-derivatives --stats

# Monitor in real-time
uv run ogm-enrich-derivatives --monitor-only

# Optional: Install rq-dashboard for web UI
uv sync --extra monitor
rq-dashboard --redis-url redis://localhost:6379
```

### Debug a Single Document

```bash
# Preview without processing
uv run ogm-enrich-derivatives --dry-run --id "stanford-abc123"

# Process single document
uv run ogm-enrich-derivatives --id "stanford-abc123"
```

## Common Gotchas

1. **Redis required** - Start with `docker compose up -d` before running derivatives
2. **macOS fork safety** - Workers set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` automatically
3. **SQLite WAL mode** - Database uses WAL for concurrent access from multiple workers
4. **ID prefixes stripped** - `stanford-abc123` becomes `abc123` in output paths
5. **Retry logic** - Failed jobs are re-enqueued with exponential backoff (max 5 retries)
6. **Status tracking** - Jobs marked `enqueued` won't be re-queued on restart
