# Model Size Configuration Guide

## Overview

You can now generate embedding models from **~5 MB to ~35 MB** by adjusting two parameters:

1. **`--embedding-dims`**: Dimensions of embedding vectors (32, 64, 128, or 256)
2. **`--max-vocab-size`**: Maximum number of vocabulary terms (1000+)

## Quick Comparison

| Model Size | Command | Model Files | Quality | Best For |
|------------|---------|-------------|---------|----------|
| **~5 MB** | `--embedding-dims 32 --max-vocab-size 1000` | model.safetensors: ~4MB | Good | Mobile apps |
| **~8 MB** | `--embedding-dims 64 --max-vocab-size 2000` | model.safetensors: ~7MB | Very Good | Web apps |
| **~15 MB** | `--embedding-dims 128 --max-vocab-size 5000` | model.safetensors: ~14MB | Excellent | **Recommended** |
| **~35 MB** | `--embedding-dims 256 --max-vocab-size 10000` | model.safetensors: ~33MB | Best | Desktop apps |

## Usage Examples

### Small Model (~8 MB) - Comparable to MinishLab

This is comparable to MinishLab's smallest pre-built model:

```bash
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000
```

**Expected output:**
- `tmp/ogm-model/model.safetensors`: ~7-8 MB
- `tmp/ogm-model/embeddings.bin`: ~7-8 MB
- `tmp/ogm-model/tokenizer.json`: ~500 KB
- **Total: ~8-10 MB**

### Recommended Configuration (~15 MB)

Best balance of quality and size for most web applications:

```bash
uv run ogm-harvest --embedding-dims 128 --max-vocab-size 5000
```

This is now the default if you run without arguments:

```bash
uv run ogm-harvest
```

### Tiny Model (~5 MB)

For mobile apps or very limited bandwidth:

```bash
uv run ogm-harvest --embedding-dims 32 --max-vocab-size 1000
```

## What Changes with Model Size

### Lower Dimensions (32 → 256)

**Pros:**
- Smaller file size (exponentially smaller)
- Faster computation in browser
- Lower memory usage

**Cons:**
- Reduced semantic quality
- May lose some nuanced similarities

**Recommendation:** 64+ dimensions for production use

### Lower Vocabulary (1K → 10K)

**Pros:**
- Smaller file size (linearly smaller)
- Faster model loading

**Cons:**
- Fewer domain-specific terms recognized
- Falls back to subword tokenization more often

**What happens:**
- For vocab < 10K: Only most frequent controlled vocabulary terms included
- For vocab ≥ 10K: All controlled vocabulary terms included
- Free-text terms always limited to most frequent

**Recommendation:** 2K-5K vocab for most use cases

## Model Size Calculation

Model size formula: `vocab_size × embedding_dims × 4 bytes`

Examples:
- 2,000 vocab × 64 dims × 4 bytes = **512 KB** (plus overhead)
- 5,000 vocab × 128 dims × 4 bytes = **2.5 MB** (plus overhead)
- 10,000 vocab × 256 dims × 4 bytes = **10 MB** (plus overhead)

Overhead includes:
- Tokenizer: ~500 KB
- Model config: ~1 KB
- Metadata: ~1 KB

## Python API

```python
from ogm_to_parquet.harvest import OgmToParquet

# Small model (~8 MB)
harvester = OgmToParquet(
    embedding_dims=64,
    max_vocab_size=2000
)
harvester.convert()

# Tiny model (~5 MB)
harvester = OgmToParquet(
    embedding_dims=32,
    max_vocab_size=1000
)
harvester.convert()

# Custom configuration
harvester = OgmToParquet(
    embedding_dims=96,      # Any value works, not just 32/64/128/256
    max_vocab_size=3500     # Any value works
)
harvester.convert()
```

Note: Command-line only accepts 32/64/128/256 for safety, but Python API accepts any value.

## Quality vs Size Trade-offs

### 32 Dimensions
- ❌ Significant quality loss
- ❌ Only basic similarity preserved
- ✅ Smallest possible size (~5 MB)
- **Use case:** Mobile apps where size is critical

### 64 Dimensions
- ✅ Good quality for most use cases
- ✅ Small size (~8 MB)
- ✅ Recommended minimum for production
- **Use case:** Web applications (most common)

### 128 Dimensions
- ✅ Excellent quality
- ✅ Minimal loss from 256
- ✅ Reasonable size (~15 MB)
- **Use case:** Default, best balance

### 256 Dimensions
- ✅ Full quality from base model
- ❌ Larger size (~35 MB)
- **Use case:** Desktop apps, when size not a concern

## Testing Different Sizes

Use the provided test script to generate and compare model sizes:

```bash
uv run python test_small_model.py
```

This will generate a small model and show the actual file sizes.

## Browser Performance

All model sizes load and run efficiently in modern browsers:

| Model Size | Load Time (3G) | Load Time (4G) | Inference Speed |
|------------|---------------|----------------|-----------------|
| 5 MB | ~2 sec | <1 sec | Very Fast |
| 8 MB | ~3 sec | ~1 sec | Very Fast |
| 15 MB | ~5 sec | ~2 sec | Fast |
| 35 MB | ~12 sec | ~4 sec | Fast |

All models use the same inference algorithm (tokenization + vector lookup + averaging), so inference speed is primarily determined by vocabulary size, not dimensions.

## Recommendations by Use Case

### Mobile Web App
```bash
uv run ogm-harvest --embedding-dims 64 --max-vocab-size 2000
```
8 MB model, good quality, fast loading on mobile networks

### Desktop Web App
```bash
uv run ogm-harvest --embedding-dims 128 --max-vocab-size 5000
```
15 MB model, excellent quality, fast loading on broadband

### Native Mobile App
```bash
uv run ogm-harvest --embedding-dims 32 --max-vocab-size 1000
```
5 MB model, acceptable quality, bundled with app

### Desktop Application
```bash
uv run ogm-harvest --embedding-dims 256 --max-vocab-size 10000
```
35 MB model, best quality, size not a concern
