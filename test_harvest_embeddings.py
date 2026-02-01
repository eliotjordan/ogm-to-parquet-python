"""Test script to debug embedding generation in harvest pipeline."""

import logging
from ogm_to_parquet.harvest import OgmToParquet

# Set up verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting harvest test...")

    # Create harvester with embeddings enabled
    harvester = OgmToParquet(
        ogm_path="./tmp/opengeometadata/",
        output_path="./tmp/test_ogm.parquet",
        model_dir="./tmp/ogm-model/",
        enable_embeddings=True,
    )

    logger.info(f"Embeddings enabled: {harvester.enable_embeddings}")
    logger.info(f"Embedding generator: {harvester.embedding_generator}")

    # Run conversion
    harvester.convert()

    logger.info(f"Conversion complete. Embedding generator after convert: {harvester.embedding_generator}")
    logger.info(f"Number of rows: {len(harvester.rows)}")

    # Check embeddings in first few rows
    for i, row in enumerate(harvester.rows[:3]):
        emb = row.get('embeddings')
        if emb is None:
            logger.warning(f"Row {i} ({row.get('id')}): embeddings = NULL")
        else:
            logger.info(f"Row {i} ({row.get('id')}): embeddings length = {len(emb)}, first 3 values = {emb[:3]}")

if __name__ == "__main__":
    main()
