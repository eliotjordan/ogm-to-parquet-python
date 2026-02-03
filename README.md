# OpenGeoMetadata to Parquet Converter (Python)

Python implementation of the OpenGeoMetadata to Parquet converter. This tool processes OpenGeoMetadata JSON files and converts them into a single Parquet file optimized for querying with DuckDB.

## Requirements

- Python 3.11+
- uv (for package management)

## Installation

```bash
# Install dependencies (includes model2vec for embedding generation)
uv sync --all-extras

# Install with distillation support (adds torch and sentence-transformers)
# Required for generating custom embedding models
uv sync --extra distill
```

## Usage

```bash
# Run with default settings (128 dims, 10K vocab, ~29MB model)
uv run ogm-harvest

# Generate a small model (~8 MB, good for web)
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000

# Generate a tiny model (~5 MB, for mobile)
uv run ogm-harvest --embedding-dims 32 --max-vocab-size 1000

# Generate a large model (~35 MB, best quality)
uv run ogm-harvest --embedding-dims 256 --max-vocab-size 10000

# Disable embeddings entirely
uv run ogm-harvest --no-embeddings

# Or run the module directly
uv run python -m ogm_to_parquet.harvest
```

### Downlaod OGM metadata

```
# Download OpenGeoMetadata repositories and harvest
uv run ogm-harvest --download

# Download using custom repos config
uv run ogm-harvest --download --repos-config my-repos.yaml

# Download to custom directory
uv run ogm-harvest --download --ogm-path ./data/opengeometadata

# Download only (skip harvesting)
uv run ogm-harvest --download-only
```

### Embedding Generation

The harvester automatically generates semantic embeddings for each document:

1. **Vocabulary Building**: Extracts vocabulary from metadata fields
   - Controlled vocabulary: creator, location, provider, resource_class, subject, theme, format
   - Free-text terms: title, description, publisher (common terms extracted)
   - Configurable size: 1K to 10K+ terms

2. **Model Distillation**: Creates a small, browser-compatible model
   - Distills `sentence-transformers/all-MiniLM-L6-v2`
   - Custom vocabulary ensures good domain coverage
   - Configurable dimensions: 32, 64, 128, or 256
   - Outputs saved to `tmp/ogm-model/`

3. **Document Embedding**: Generates embedding vectors for each record
   - Saved in `embeddings` field in Parquet file
   - Suitable for semantic search in DuckDB or browser

### Model Sizes

| Configuration | Model Size | Quality | Use Case |
|--------------|------------|---------|----------|
| 32 dims, 1K vocab | ~5 MB | Good | Mobile apps |
| 64 dims, 2K vocab | ~8 MB | Very Good | Web apps (recommended) |
| 128 dims, 5K vocab | ~15 MB | Excellent | Default |
| 256 dims, 10K vocab | ~35 MB | Best | Desktop apps |

See `examples_model_sizes.md` for detailed configuration examples.

### Disabling Embeddings

To skip embedding generation:

```python
from ogm_to_parquet.harvest import OgmToParquet

harvester = OgmToParquet(enable_embeddings=False)
harvester.convert()
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_geometry.py

# Lint code
uv run ruff check src tests

# Format code
uv run ruff format src tests
```

### Continuous Integration

The project uses GitHub Actions for CI:
- **Linting**: Runs ruff linter and formatter checks
- **Testing**: Runs full test suite on Python 3.11 and 3.12 with coverage
- **Compatibility**: Verifies tests work without optional distillation dependencies

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
2. Builds custom vocabulary from all documents (controlled vocab + extracted terms)
3. Distills embedding model with custom vocabulary (saved to `tmp/ogm-model/`)
4. For each document:
   - Transforms field names from GeoBlacklight schema to simplified schema
   - Converts geometries from WKT/ENVELOPE to GeoJSON
   - Extracts thumbnail URLs from references field
   - Generates 256-dimensional embedding vector
5. Writes data to `tmp/ogm.parquet` with ZSTD compression

The generated Parquet file includes:
- **geojson** (string): GeoJSON text representation for compatibility
- **embeddings** (float32[]): 256-dim embedding vectors for semantic search
- All standard metadata fields (title, creator, subject, etc.)

## Testing

The project includes comprehensive tests with pytest and coverage reporting:

- **74 tests total** covering geometry, harvest, and embeddings modules
- **High overall coverage**
  - geometry.py: 98% coverage
  - harvest.py: 93% coverage
  - embeddings.py: 60% coverage (core logic tested; distillation tests optional)

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

# Run distillation tests (requires torch and sentence-transformers)
uv sync --extra distill
uv run pytest tests/test_embeddings.py
```

## Output Files

### Parquet Data (`tmp/ogm.parquet`)

The harvester generates a single Parquet file with all metadata and embeddings:
- **geojson**: Text GeoJSON for display and compatibility
- **embeddings**: 256-dim float32 vectors for semantic search
- All standard metadata fields

### Embedding Model (`tmp/ogm-model/`)

The distilled Model2Vec model for browser use:
- **tokenizer.json**: HuggingFace tokenizer (load with tokenizers.js)
- **embeddings.safetensors**: Embedding matrix in safetensors format
- **embeddings.bin**: Raw float32 binary (simpler browser loading)
- **metadata.json**: Vocab size, embedding dimensions, base model info

### Using with DuckDB

To query with DuckDB spatial functions:
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

## Using Embeddings in the Browser

The distilled model can be loaded in JavaScript for client-side semantic search:

```javascript
import { Tokenizer } from "@huggingface/tokenizers";

// Load model assets (once on startup)
const tokenizer = await Tokenizer.from_pretrained("./tmp/ogm-model/tokenizer.json");

// Load embedding matrix as Float32Array
const response = await fetch("./tmp/ogm-model/embeddings.bin");
const buffer = await response.arrayBuffer();
const embeddingMatrix = new Float32Array(buffer);
const EMBEDDING_DIM = 256;

// Generate query embedding
function encodeQuery(text) {
  const encoded = tokenizer.encode(text);
  const tokenIds = encoded.ids;

  // Average token embeddings
  const sum = new Float32Array(EMBEDDING_DIM);
  for (const id of tokenIds) {
    const offset = id * EMBEDDING_DIM;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      sum[i] += embeddingMatrix[offset + i];
    }
  }

  // Normalize
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    sum[i] /= tokenIds.length;
  }

  return sum;
}

// Compute cosine similarity with document embeddings
const queryEmbedding = encodeQuery("water resources california");
// Compare with document embeddings from Parquet file
```

No neural network inference required - just tokenization and vector math. The entire model loads in milliseconds and runs efficiently in the browser.

## Differences from Ruby Version

This Python implementation extends the Ruby version with:

1. **Semantic embedding generation** using Model2Vec (new feature)
2. Comprehensive test suite (74 tests vs 0)
3. Type hints throughout for better IDE support
4. Modern package management with uv
5. Better error handling and logging
6. Easier cross-platform setup
7. Well-documented code with docstrings
