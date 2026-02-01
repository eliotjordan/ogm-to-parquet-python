# Model Size Examples

The harvester supports generating different model sizes by adjusting two parameters:

1. **embedding_dims**: Number of dimensions in the embedding vectors (32, 64, 128, or 256)
2. **max_vocab_size**: Maximum number of vocabulary terms to include

## Model Size Trade-offs

| Configuration | Model Size | Use Case |
|--------------|------------|----------|
| **Tiny** (32 dims, 1K vocab) | ~3-5 MB | Mobile apps, very limited bandwidth |
| **Small** (64 dims, 2K vocab) | ~5-8 MB | Web apps with strict size requirements |
| **Medium** (128 dims, 5K vocab) | ~12-18 MB | Balanced quality/size (recommended) |
| **Large** (256 dims, 10K vocab) | ~30-40 MB | Best quality, desktop apps |

## Usage Examples

### Tiny Model (~5 MB)
Best for mobile or severely bandwidth-constrained environments. Lower quality but very fast.

```bash
uv run ogm-harvest --embedding-dims 32 --max-vocab-size 1000
```

### Small Model (~8 MB)
Good balance for web applications. Comparable to MinishLab's smallest model.

```bash
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000
```

### Medium Model (~15 MB) - Recommended
Best balance of quality and size for most web applications.

```bash
uv run ogm-harvest --embedding-dims 128 --max-vocab-size 5000
```

### Large Model (~35 MB)
Best quality, suitable when file size is not a concern.

```bash
uv run ogm-harvest --embedding-dims 256 --max-vocab-size 10000
```

### Custom Configuration
You can mix and match dimensions and vocabulary size:

```bash
# High quality, smaller vocab (good for specialized domains)
uv run ogm-harvest --embedding-dims 256 --max-vocab-size 3000

# Lower dimensions, larger vocab (good for broad domains)
uv run ogm-harvest --embedding-dims 128 --max-vocab-size 8000
```

## Python API

```python
from ogm_to_parquet.harvest import OgmToParquet

# Tiny model
harvester = OgmToParquet(
    embedding_dims=32,
    max_vocab_size=1000
)
harvester.convert()

# Small model (recommended for web)
harvester = OgmToParquet(
    embedding_dims=64,
    max_vocab_size=2000
)
harvester.convert()

# Medium model (best balance)
harvester = OgmToParquet(
    embedding_dims=128,
    max_vocab_size=5000
)
harvester.convert()
```

## What Gets Reduced

When you reduce the model size:

1. **Lower embedding_dims**:
   - Smaller embedding vectors (e.g., 64 floats instead of 256)
   - Faster computation in browser
   - Slightly lower semantic similarity quality

2. **Lower max_vocab_size**:
   - Fewer vocabulary terms included
   - Only most frequent controlled vocabulary terms
   - Fewer extracted terms from free-text fields
   - Model can still handle any text (falls back to subword tokenization)

## Model Size Breakdown

For a model with **64 dims** and **2000 vocab**:

```
tokenizer.json:        ~500 KB  (HuggingFace tokenizer)
model.safetensors:     ~4-6 MB  (embedding matrix: vocab_size × dims × 4 bytes)
embeddings.bin:        ~4-6 MB  (same as safetensors, raw binary)
metadata.json:         ~500 B   (model metadata)
config.json:           ~500 B   (model config)
-------------------------------------------
Total:                 ~8-12 MB
```

## Quality Considerations

- **32 dimensions**: Significant quality loss, but usable for basic similarity
- **64 dimensions**: Good quality for most use cases, recommended minimum
- **128 dimensions**: High quality, minimal loss from 256
- **256 dimensions**: Full quality from base model

The vocabulary size mainly affects coverage of domain-specific terms. With subword fallback, even small vocabularies can handle arbitrary text.
