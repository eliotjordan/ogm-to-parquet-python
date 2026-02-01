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
4. Creates native geometry field in WKB (Well-Known Binary) format
5. Extracts thumbnail URLs from references field
6. Writes data to `tmp/ogm.parquet` with ZSTD compression

The generated Parquet file includes both:
- **geojson** (string): GeoJSON text representation for compatibility
- **geometry** (binary): WKB-encoded geometry for efficient spatial operations

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

## Geometry Support

The harvester automatically creates a native geometry field in WKB (Well-Known Binary) format. No post-processing is needed!

The Parquet file includes:
- **geojson**: Text GeoJSON for display and compatibility
- **geometry**: Binary WKB for efficient spatial queries with DuckDB

To use with DuckDB spatial functions:
```sql
-- Query by bounding box
SELECT * FROM 'ogm.parquet'
WHERE ST_Intersects(
  ST_GeomFromWKB(geometry),
  ST_MakeEnvelope(-122.5, 37.7, -122.0, 37.8)
);

-- Or use the geojson field
SELECT * FROM 'ogm.parquet'
WHERE ST_Intersects(
  ST_GeomFromGeoJSON(geojson),
  ST_MakeEnvelope(-122.5, 37.7, -122.0, 37.8)
);
```

## Differences from Ruby Version

This Python implementation is functionally equivalent to the Ruby version but includes:

1. Comprehensive test suite (94% coverage)
2. Type hints throughout for better IDE support
3. Modern package management with uv
4. Better error handling and logging
5. Easier cross-platform setup
6. Well-documented code with docstrings
