# OpenGeoMetadata to Parquet Converter (Python)

Python implementation of the OpenGeoMetadata to Parquet converter. This tool processes OpenGeoMetadata JSON files and converts them into a single Parquet file optimized for querying with DuckDB.

## Requirements

- Python 3.11+
- uv (for package management)

## Installation

```bash
# Install dependencies
uv sync --all-extras
```

## Usage

```bash
# Run the harvester
uv run python -m ogm_to_parquet.harvest

# Or with uv directly
uv run harvest
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_geometry.py
```

## Project Structure

```
src/ogm_to_parquet/
├── __init__.py          # Package initialization
├── geometry.py          # Geometry transformation utilities
└── harvest.py           # Main harvester script

tests/
├── test_geometry.py     # Geometry module tests
└── test_harvest.py      # Harvest module tests
```

## How It Works

1. Reads OpenGeoMetadata JSON files from `tmp/opengeometadata/`
2. Transforms field names from GeoBlacklight schema to simplified schema
3. Converts geometries from WKT/ENVELOPE to GeoJSON
4. Extracts thumbnail URLs from references field
5. Writes data to `tmp/ogm.parquet` with ZSTD compression

## Testing

The project includes comprehensive tests with pytest and coverage reporting:

- **40 tests total** covering geometry and harvest modules
- **94% overall coverage**
  - geometry.py: 98% coverage
  - harvest.py: 93% coverage

```bash
# Run tests
uv run pytest

# Run with coverage report
uv run pytest --cov

# Generate HTML coverage report
uv run pytest --cov-report=html

# Run specific test file
uv run pytest tests/test_geometry.py

# Run tests matching pattern
uv run pytest -k "envelope"
```

## Post-Processing with DuckDB

After generating `ogm.parquet`, add a native geometry column:

```bash
duckdb -c "
COPY (
  SELECT *,
    ST_GeomFromGeoJSON(geojson) AS geometry
  FROM 'tmp/ogm.parquet'
) TO 'tmp/cloud.parquet' (FORMAT PARQUET, COMPRESSION zstd, PARQUET_VERSION v2);
"
```

This creates `cloud.parquet` with WKB geometry for efficient spatial queries.

## Differences from Ruby Version

This Python implementation is functionally equivalent to the Ruby version but includes:

1. Comprehensive test suite (94% coverage)
2. Type hints throughout for better IDE support
3. Modern package management with uv
4. Better error handling and logging
5. Easier cross-platform setup
6. Well-documented code with docstrings
