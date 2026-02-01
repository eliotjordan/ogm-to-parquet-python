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

The project includes comprehensive tests with pytest and coverage reporting. Target coverage is 80%+.
