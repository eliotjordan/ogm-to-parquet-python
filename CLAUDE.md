# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python tool that converts OpenGeoMetadata JSON files into a single Parquet file optimized for querying with DuckDB. This is the modern Python rewrite of the Ruby harvester (located in sibling directory `../ogm-to-parquet/`).

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Install with distillation support (includes torch and sentence-transformers)
uv sync --extra distill

# Run harvester with default settings (128 dims, 5K vocab, ~15MB model)
uv run ogm-harvest

# Run with small model (~8 MB, good for web)
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000

# Run with tiny model (~5 MB, for mobile)
uv run ogm-harvest --embedding-dims 32 --max-vocab-size 1000

# Run without embeddings
uv run ogm-harvest --no-embeddings

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_geometry.py

# Run tests matching pattern
uv run pytest -k "envelope"

# Run distillation tests (requires --extra distill)
uv run pytest tests/test_embeddings.py -k "distill"

# Lint code with ruff
uv run ruff check src tests

# Format code with ruff
uv run ruff format src tests

# Check formatting without making changes
uv run ruff format --check src tests
```

## Architecture

### Core Components

**src/ogm_to_parquet/embeddings.py** - Embedding generation with Model2Vec:
- `VocabularyBuilder` extracts vocabulary from controlled vocab and free-text fields
- `distill_model()` distills all-MiniLM-L6-v2 with custom vocabulary for browser use
- `EmbeddingGenerator` generates document embeddings using the distilled model
- Outputs tokenizer.json, model.safetensors, embeddings.bin, and metadata.json

**src/ogm_to_parquet/harvest.py** - Main harvester orchestration:
- `OgmToParquet` class manages the entire conversion pipeline
- Recursively scans `tmp/opengeometadata/` for JSON files
- Field mapping via `FIELD_MAP` (GeoBlacklight schema → simplified schema)
- Data cleaning removes single quotes to prevent SQL injection
- PyArrow schema in `PARQUET_SCHEMA` defines output structure
- Outputs to `tmp/ogm.parquet` with ZSTD compression

**src/ogm_to_parquet/geometry.py** - Geometry transformations:
- `Geometry` class converts WKT and ENVELOPE syntax to GeoJSON
- Uses Shapely for WKT parsing and bounding box extraction
- Provides world extent fallback for invalid geometries
- ENVELOPE format: `ENVELOPE(minx, maxx, maxy, miny)` → WKT POLYGON

### Data Pipeline

1. **Collection** (`_collect_documents`): Recursively find all `.json` files under `tmp/opengeometadata/`
2. **Vocabulary Building** (`_prepare_embedding_model`): Extract vocabulary from all documents
   - Controlled vocabulary fields added directly (creator, location, provider, resource_class, etc.)
   - Free-text fields tokenized and common terms extracted (title, description, publisher)
   - Bigrams included for domain-specific phrases
3. **Model Distillation**: Distill all-MiniLM-L6-v2 with custom vocabulary
   - PCA reduction to 256 dimensions for small file size
   - Outputs saved to `tmp/ogm-model/` (tokenizer.json, model.safetensors, embeddings.bin, metadata.json)
   - Metadata saved for browser loading
4. **Document Processing**: For each document:
   - **Remapping** (`_remap_doc_keys`): Apply `FIELD_MAP` to convert GeoBlacklight field names
   - **Cleaning** (`_clean_values`): Strip single quotes from strings, preserve numeric types
   - **Embedding Generation**: Generate 256-dim embedding vector using distilled model
   - **Row Building** (`_build_row`): Extract geometry, thumbnail, embeddings, ensure correct types
5. **Parquet Writing** (`_write_parquet`): Convert to PyArrow Table and write with ZSTD compression

### Field Types

**Scalar fields** (pa.string()): `id`, `title`, `provider`, `access_rights`, `format`, `thumbnail`, `geojson`, `description`, `wxs_identifier`, `modified`

**List fields** (pa.list_(pa.string())): `creator`, `location`, `publisher`, `resource_class`, `resource_type`, `subject`, `theme`, `identifier`, `temporal`

**Float list field**: `index_year` (pa.list_(pa.float64())) - converts strings/ints to floats

**Embedding field**: `embeddings` (pa.list_(pa.float32())) - 256-dimensional embedding vectors for semantic search

**Important**: List fields in Parquet become DuckDB LIST columns requiring `list_contains()` or `UNNEST()` for querying.

### Geometry Handling

The harvester generates GeoJSON from the `dcat_bbox` field. The output includes:
- **geojson** (string): Text GeoJSON representation
- No native geometry field (removed in recent update - was generating WKB but caused issues)

If native DuckDB GEOMETRY type is needed, use `convert_geometry.sql` post-processing script:
```sql
INSTALL spatial;
LOAD spatial;
COPY (
  SELECT *, ST_GeomFromGeoJSON(geojson) AS geometry
  FROM 'tmp/ogm.parquet'
) TO 'tmp/cloud.parquet' (FORMAT PARQUET, COMPRESSION zstd, PARQUET_VERSION v2);
```

### Thumbnail Extraction

Parses `dct_references_s` JSON field (harvest.py:343-373):
1. Prefers `http://schema.org/thumbnailUrl`
2. Falls back to `http://iiif.io/api/image`, converting `info.json` URL to 150x150 thumbnail
3. Returns None if neither available

### Type Coercion Helpers

**_ensure_string**: Converts to string, joins list elements with space
**_ensure_list**: Wraps scalars in list, returns None for empty lists
**_ensure_float_list**: Converts strings/ints to floats, filters non-convertible values

## Testing Strategy

**74 tests total** with extensive coverage:
- **test_geometry.py**: 14 tests for WKT/ENVELOPE parsing, GeoJSON conversion, bbox extraction
- **test_harvest.py**: 42 tests for field mapping, data cleaning, thumbnail extraction, type coercion
- **test_embeddings.py**: 18 tests for vocabulary building, embedding generation, model distillation
  - 14 tests always run (vocabulary building, tokenization logic)
  - 4 tests require torch/sentence-transformers (marked with `@requires_distill`, skipped unless `uv sync --extra distill`)

### Running Specific Tests

```bash
# Single test file
uv run pytest tests/test_geometry.py

# Single test method
uv run pytest tests/test_harvest.py::TestOgmToParquet::test_remap_doc_keys

# Tests matching pattern
uv run pytest -k "thumbnail"

# Verbose output
uv run pytest -v

# With coverage report
uv run pytest --cov

# Generate HTML coverage report
uv run pytest --cov-report=html
```

### Coverage Configuration

Coverage thresholds defined in `pyproject.toml`:
- Source: `src/ogm_to_parquet`
- Excludes: `tests/*`, pragma comments, `__repr__`, `__main__`
- Reports: terminal + HTML

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

**Configuration** (in `pyproject.toml`):
- Line length: 100 characters
- Target: Python 3.11+
- Enabled rules: pycodestyle, pyflakes, isort, flake8-bugbear, flake8-comprehensions, pyupgrade
- Auto-formatting with double quotes and spaces

**Commands**:
```bash
# Check linting
uv run ruff check src tests

# Auto-fix linting issues
uv run ruff check --fix src tests

# Check formatting
uv run ruff format --check src tests

# Apply formatting
uv run ruff format src tests
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push to main and pull requests:

**Lint Job**:
- Runs ruff linter and formatter checks
- Python 3.11 on Ubuntu

**Test Job**:
- Runs full test suite with coverage on Python 3.11 and 3.12
- Includes distillation tests (with torch/sentence-transformers)
- Uploads coverage to Codecov

**Test-No-Distill Job**:
- Verifies tests work without optional distillation dependencies
- Ensures core functionality doesn't require torch
- Distillation tests are automatically skipped

## Adding New Fields

1. Add mapping to `FIELD_MAP` in harvest.py (line 24)
2. Add field to `PARQUET_SCHEMA` with correct PyArrow type (line 48)
3. Update `_build_row()` method with field extraction logic (line 218)
4. Add tests in `tests/test_harvest.py`
5. Update downstream consumers (e.g., `cloud-ogm-react/src/lib/fieldsConfig.ts`)

## Data Cleaning

All string values have single quotes stripped (`value.replace("'", "")`) to prevent SQL injection when the Parquet file is queried. This applies to:
- Scalar string fields
- Array/list elements that are strings
- Does NOT apply to numeric types (preserved as-is)

## Key Differences from Ruby Version

1. Comprehensive test suite (56 tests vs 0)
2. Type hints throughout for IDE support
3. Modern package management with uv
4. Better error handling and logging
5. No SQLite dependency (Ruby version uses geo_combine)
6. Simpler geometry handling (Shapely instead of RGeo)

## Embedding Generation

### Vocabulary Building

The harvester builds a custom vocabulary from metadata fields:

**Controlled Vocabulary Fields** (added directly):
- creator, location, provider, access_rights, resource_class, resource_type, subject, theme, format

**Free-Text Fields** (common terms extracted):
- title, description, publisher

**Process**:
1. All controlled vocab values are lowercased and added to vocabulary
2. Free-text fields are tokenized and filtered:
   - Stopwords removed (the, a, an, of, etc.)
   - Minimum term frequency threshold (default: 2)
   - Top N most common terms extracted (default: 10,000)
   - Bigrams included for domain-specific phrases

### Model Distillation

Uses Model2Vec to distill `sentence-transformers/all-MiniLM-L6-v2`:
- PCA reduction to configurable dimensions (32, 64, 128, or 256)
- Custom vocabulary ensures good coverage of domain terms
- Configurable vocabulary size (1K to 10K+ terms)
- Output files saved to `tmp/ogm-model/`:
  - `tokenizer.json` - HuggingFace tokenizer (for browser use)
  - `model.safetensors` - Model weights in safetensors format
  - `embeddings.bin` - Raw float32 binary (easier to load in browser)
  - `metadata.json` - Vocab size, embedding dims, base model info

**Model Size Options:**

| Configuration | Model Size | Vocabulary | Quality |
|--------------|------------|------------|---------|
| 32 dims, 1K vocab | ~5 MB | Most frequent 1000 terms | Good |
| 64 dims, 2K vocab | ~8 MB | Most frequent 2000 terms | Very Good |
| 128 dims, 5K vocab | ~15 MB | Most frequent 5000 terms | Excellent (default) |
| 256 dims, 10K vocab | ~35 MB | 10000+ terms | Best |

For smaller models (vocab < 10K), controlled vocabulary is limited to most frequent terms only.

### Browser Loading

The distilled model can be loaded in JavaScript/WASM:
1. Load `tokenizer.json` using HuggingFace tokenizers.js
2. Load `embeddings.bin` as Float32Array
3. Tokenize query text, lookup embeddings, average vectors
4. No neural network required - just tokenization and vector arithmetic

See embeddings.py docstring for implementation details.

### Disabling Embeddings

Pass `enable_embeddings=False` to `OgmToParquet` constructor:
```python
harvester = OgmToParquet(enable_embeddings=False)
harvester.convert()
```

## Dependencies

**Core** (pyproject.toml:6-10):
- `pyarrow>=18.0.0` - Parquet file writing
- `shapely>=2.0.0` - Geometry transformations
- `geojson>=3.0.0` - GeoJSON validation
- `model2vec>=0.3.0` - Static embedding model generation

**Dev** (pyproject.toml:17-21):
- `pytest>=8.0.0` - Testing framework
- `pytest-cov>=6.0.0` - Coverage reporting
- `ruff>=0.8.0` - Linting and formatting

**Distill** (optional, pyproject.toml:20-23):
- `torch>=2.0.0` - PyTorch for model distillation
- `sentence-transformers>=2.2.0` - Base model for distillation
- Only needed if running distillation (not needed for inference)

## Common Gotchas

1. ENVELOPE format uses order `(minx, maxx, maxy, miny)` - note that maxy comes before miny
2. Invalid geometries return world extent `-180,-90,180,90` instead of failing
3. Empty lists become None in Parquet schema (not empty arrays)
4. The `references` field must be valid JSON string (not a dict)
5. `index_year` field converts strings to floats, skips non-convertible values
6. Test coverage configuration requires `--cov` flag to generate reports
