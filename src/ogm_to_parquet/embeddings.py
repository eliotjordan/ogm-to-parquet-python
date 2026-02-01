"""Embedding generation using Model2Vec for OpenGeoMetadata documents.

Builds a custom vocabulary from metadata fields (both controlled vocabulary and
extracted free-text terms), distills a small embedding model, and generates
document embeddings suitable for browser-based semantic search.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Fields to include in vocabulary, by type
CONTROLLED_VOCAB_FIELDS = [
    "creator",
    "location",
    "provider",
    "access_rights",
    "resource_class",
    "resource_type",
    "subject",
    "theme",
    "format",
]

FREE_TEXT_FIELDS = [
    "title",
    "description",
    "publisher",
]

# Simple stopwords for term extraction (extend as needed)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
}


class VocabularyBuilder:
    """Builds vocabulary from OpenGeoMetadata documents."""

    def __init__(
        self,
        max_vocab_size: int = 10000,
        min_term_freq: int = 2,
        include_bigrams: bool = True,
    ):
        """Initialize vocabulary builder.

        Args:
            max_vocab_size: Maximum number of terms to extract from free-text fields
            min_term_freq: Minimum frequency for a term to be included
            include_bigrams: Whether to include bigrams (2-word phrases)
        """
        self.max_vocab_size = max_vocab_size
        self.min_term_freq = min_term_freq
        self.include_bigrams = include_bigrams
        self.vocab: Set[str] = set()

    def build_vocabulary(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Build vocabulary from all documents.

        Args:
            documents: List of metadata documents

        Returns:
            List of vocabulary terms
        """
        logger.info(f"Building vocabulary from {len(documents)} documents")

        # Collect controlled vocabulary terms
        controlled_terms = self._extract_controlled_vocab(documents)
        logger.info(f"Extracted {len(controlled_terms)} controlled vocabulary terms")

        # Extract domain-specific terms from free-text fields
        free_text_terms = self._extract_free_text_terms(documents)
        logger.info(f"Extracted {len(free_text_terms)} terms from free-text fields")

        # Combine and deduplicate
        self.vocab = controlled_terms | free_text_terms
        vocab_list = sorted(list(self.vocab))

        logger.info(f"Final vocabulary size: {len(vocab_list)} terms")
        return vocab_list

    def _extract_controlled_vocab(
        self, documents: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract all unique values from controlled vocabulary fields.

        Args:
            documents: List of metadata documents

        Returns:
            Set of controlled vocabulary terms
        """
        terms = set()

        for doc in documents:
            for field in CONTROLLED_VOCAB_FIELDS:
                value = doc.get(field)
                if value is None:
                    continue

                # Handle both single values and lists
                if isinstance(value, list):
                    for item in value:
                        if item and isinstance(item, str):
                            terms.add(item.strip().lower())
                elif isinstance(value, str) and value:
                    terms.add(value.strip().lower())

        return terms

    def _extract_free_text_terms(
        self, documents: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract common terms from free-text fields.

        Uses frequency analysis to identify domain-specific terms.

        Args:
            documents: List of metadata documents

        Returns:
            Set of extracted terms
        """
        # Collect all free-text content
        all_text = []
        for doc in documents:
            parts = []
            for field in FREE_TEXT_FIELDS:
                value = doc.get(field)
                if value:
                    if isinstance(value, list):
                        parts.extend([str(v) for v in value if v])
                    elif isinstance(value, str):
                        parts.append(value)
            if parts:
                all_text.append(" ".join(parts))

        if not all_text:
            logger.warning("No free-text content found")
            return set()

        # Extract unigrams and optionally bigrams
        unigrams = Counter()
        bigrams = Counter()

        for text in all_text:
            tokens = self._tokenize(text)

            # Count unigrams
            for token in tokens:
                if token not in STOPWORDS and len(token) > 2:
                    unigrams[token] += 1

            # Count bigrams if enabled
            if self.include_bigrams and len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    if tokens[i] not in STOPWORDS or tokens[i + 1] not in STOPWORDS:
                        bigram = f"{tokens[i]} {tokens[i + 1]}"
                        bigrams[bigram] += 1

        # Filter by minimum frequency
        unigrams = {
            term: count
            for term, count in unigrams.items()
            if count >= self.min_term_freq
        }
        bigrams = {
            term: count
            for term, count in bigrams.items()
            if count >= self.min_term_freq
        }

        # Take top N most common terms
        top_unigrams = [
            term for term, _ in Counter(unigrams).most_common(self.max_vocab_size)
        ]
        top_bigrams = [
            term
            for term, _
            in Counter(bigrams).most_common(self.max_vocab_size // 2)
        ]

        return set(top_unigrams + top_bigrams)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        text = text.lower()
        # Split on non-alphanumeric, keep tokens with letters
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens


class EmbeddingGenerator:
    """Generates embeddings for documents using a distilled Model2Vec model."""

    def __init__(self, model_path: str):
        """Initialize generator with a saved model.

        Args:
            model_path: Path to directory containing saved model files
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the distilled model."""
        try:
            from model2vec import StaticModel

            self.model = StaticModel.from_pretrained(str(self.model_path))
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_embedding(self, doc: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for a single document.

        Combines multiple metadata fields into a single text representation,
        then generates an embedding vector.

        Args:
            doc: Document with metadata fields

        Returns:
            Embedding vector as list of floats, or None on error
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None

        # Construct text from document fields
        text = self._doc_to_text(doc)
        if not text:
            logger.debug(f"No text content for document {doc.get('id', 'unknown')}")
            return None

        try:
            # Generate embedding (returns numpy array)
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
            return None

    def _doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text representation for embedding.

        Args:
            doc: Document dictionary

        Returns:
            Combined text representation
        """
        parts = []

        # Add title (most important)
        title = doc.get("title")
        if title:
            parts.append(str(title))

        # Add description
        description = doc.get("description")
        if description:
            if isinstance(description, list):
                parts.extend([str(d) for d in description if d])
            else:
                parts.append(str(description))

        # Add controlled vocabulary fields
        for field in CONTROLLED_VOCAB_FIELDS:
            value = doc.get(field)
            if value:
                if isinstance(value, list):
                    parts.extend([str(v) for v in value if v])
                else:
                    parts.append(str(value))

        # Add publisher and other free-text
        for field in FREE_TEXT_FIELDS:
            if field in ["title", "description"]:
                continue  # Already added
            value = doc.get(field)
            if value:
                if isinstance(value, list):
                    parts.extend([str(v) for v in value if v])
                else:
                    parts.append(str(value))

        return " ".join(parts)


def distill_model(
    vocabulary: List[str],
    output_dir: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    pca_dims: int = 256,
) -> Path:
    """Distill a Model2Vec model with custom vocabulary.

    Args:
        vocabulary: List of vocabulary terms
        output_dir: Directory to save the distilled model
        base_model: Base sentence transformer model to distill
        pca_dims: Target dimensionality for PCA reduction

    Returns:
        Path to saved model directory
    """
    try:
        from model2vec.distill import distill

        logger.info(f"Distilling model from {base_model}")
        logger.info(f"Vocabulary size: {len(vocabulary)} terms")
        logger.info(f"Target dimensions: {pca_dims}")

        # Distill the model with custom vocabulary
        model = distill(
            model_name=base_model,
            vocabulary=vocabulary,
            pca_dims=pca_dims,
        )

        # Save model files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))

        logger.info(f"Model saved to {output_path}")

        # Also export embeddings as raw binary for easier browser loading
        embeddings_array = model.embedding.astype(np.float32)
        binary_path = output_path / "embeddings.bin"
        embeddings_array.tofile(binary_path)

        # Save metadata about the embeddings
        metadata = {
            "vocab_size": embeddings_array.shape[0],
            "embedding_dim": embeddings_array.shape[1],
            "base_model": base_model,
            "pca_dims": pca_dims,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Embedding matrix saved to {binary_path}")
        logger.info(f"Shape: {embeddings_array.shape}")

        return output_path

    except ImportError:
        logger.error("model2vec not installed. Run: uv sync --all-extras")
        raise
    except Exception as e:
        logger.error(f"Failed to distill model: {e}")
        raise
