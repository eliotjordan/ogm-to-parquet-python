"""Test script to generate a small model (~8 MB) similar to MinishLab's smallest model."""

import logging
from ogm_to_parquet.harvest import OgmToParquet
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

def get_dir_size(path):
    """Calculate total size of directory in MB."""
    total = 0
    for file in Path(path).rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total / (1024 * 1024)  # Convert to MB

def test_small_model():
    """Generate a small model comparable to MinishLab's 8MB model."""

    model_dir = "./tmp/ogm-model-small/"

    # Clean up existing model
    if Path(model_dir).exists():
        shutil.rmtree(model_dir)

    logger.info("=" * 60)
    logger.info("Testing SMALL model configuration")
    logger.info("Target: ~8 MB (comparable to MinishLab's smallest model)")
    logger.info("Configuration: 64 dims, 2000 vocab")
    logger.info("=" * 60)

    # Create harvester with small model configuration
    harvester = OgmToParquet(
        ogm_path="./tmp/opengeometadata/",
        output_path="./tmp/test_small.parquet",
        model_dir=model_dir,
        enable_embeddings=True,
        embedding_dims=64,      # Smaller dimensions
        max_vocab_size=2000,    # Smaller vocabulary
    )

    logger.info("Starting harvest with small model...")
    harvester.convert()

    # Calculate model size
    if Path(model_dir).exists():
        model_size = get_dir_size(model_dir)
        logger.info("=" * 60)
        logger.info(f"MODEL SIZE: {model_size:.1f} MB")
        logger.info("=" * 60)

        # Show file breakdown
        logger.info("\nFile sizes:")
        for file in sorted(Path(model_dir).rglob('*')):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  {file.name:25s} {size_mb:>8.2f} MB")
    else:
        logger.error("Model directory not created")

if __name__ == "__main__":
    test_small_model()
