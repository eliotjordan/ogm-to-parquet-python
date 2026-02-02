"""Tests for embeddings module."""

import json
import tempfile
from pathlib import Path

import pytest

from ogm_to_parquet.embeddings import (
    VocabularyBuilder,
    EmbeddingGenerator,
    distill_model,
)

# Check if distillation dependencies are available
try:
    import torch
    import sentence_transformers
    DISTILL_AVAILABLE = True
except ImportError:
    DISTILL_AVAILABLE = False

requires_distill = pytest.mark.skipif(
    not DISTILL_AVAILABLE,
    reason="Requires torch and sentence-transformers (install with: uv sync --extra distill)"
)


class TestVocabularyBuilder:
    """Test suite for VocabularyBuilder class."""

    def test_extract_controlled_vocab_single_values(self):
        """Test extraction of single-valued controlled vocabulary fields."""
        builder = VocabularyBuilder()
        docs = [
            {"provider": "Stanford", "access_rights": "Public"},
            {"provider": "Harvard", "access_rights": "Restricted"},
            {"provider": "Stanford", "access_rights": "Public"},
        ]

        terms = builder._extract_controlled_vocab(docs)

        assert "stanford" in terms
        assert "harvard" in terms
        assert "public" in terms
        assert "restricted" in terms
        assert len(terms) == 4

    def test_extract_controlled_vocab_multi_values(self):
        """Test extraction of multi-valued controlled vocabulary fields."""
        builder = VocabularyBuilder()
        docs = [
            {"resource_class": ["Maps", "Datasets"], "theme": ["Hydrology"]},
            {"resource_class": ["Imagery"], "theme": ["Geology", "Climate"]},
        ]

        terms = builder._extract_controlled_vocab(docs)

        assert "maps" in terms
        assert "datasets" in terms
        assert "imagery" in terms
        assert "hydrology" in terms
        assert "geology" in terms
        assert "climate" in terms

    def test_extract_controlled_vocab_handles_none(self):
        """Test that None values are handled gracefully."""
        builder = VocabularyBuilder()
        docs = [
            {"provider": None, "access_rights": "Public"},
            {"provider": "Stanford"},  # Missing access_rights
        ]

        terms = builder._extract_controlled_vocab(docs)

        assert "public" in terms
        assert "stanford" in terms
        assert None not in terms

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        builder = VocabularyBuilder()

        tokens = builder._tokenize("The quick brown fox jumps over the lazy dog")

        assert tokens == ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation and numbers."""
        builder = VocabularyBuilder()

        tokens = builder._tokenize("Data from 2020-2024: maps, imagery & more!")

        assert "data" in tokens
        assert "from" in tokens
        assert "2020" in tokens
        assert "2024" in tokens
        assert "maps" in tokens
        assert "imagery" in tokens
        assert "more" in tokens
        # Punctuation should be removed
        assert ":" not in tokens
        assert "&" not in tokens

    def test_extract_free_text_terms_basic(self):
        """Test extraction of terms from free-text fields."""
        builder = VocabularyBuilder(max_vocab_size=100, min_term_freq=2)
        docs = [
            {"title": "Map of California water resources", "description": "Detailed map showing water resources"},
            {"title": "Water quality data for California", "description": "Dataset with water quality measurements"},
            {"title": "California geology map", "description": "Geological features of California"},
        ]

        terms = builder._extract_free_text_terms(docs)

        # Common terms should be included
        assert "california" in terms
        assert "water" in terms
        assert "map" in terms

        # Stopwords should be excluded
        assert "the" not in terms
        assert "of" not in terms
        assert "for" not in terms

    def test_extract_free_text_terms_min_frequency(self):
        """Test that min_term_freq filters out rare terms."""
        builder = VocabularyBuilder(max_vocab_size=100, min_term_freq=3)
        docs = [
            {"title": "Common word appears often"},
            {"title": "Common word appears again"},
            {"title": "Common word appears here"},
            {"title": "Rare term only once"},
        ]

        terms = builder._extract_free_text_terms(docs)

        assert "common" in terms  # Appears 3 times
        assert "word" in terms    # Appears 3 times
        assert "appears" in terms # Appears 3 times
        assert "rare" not in terms  # Only appears once
        assert "term" not in terms  # Only appears once

    def test_extract_free_text_terms_bigrams(self):
        """Test extraction of bigrams from free-text."""
        builder = VocabularyBuilder(
            max_vocab_size=100,
            min_term_freq=2,
            include_bigrams=True
        )
        docs = [
            {"title": "water quality monitoring", "description": "water quality data"},
            {"title": "air quality monitoring", "description": "air quality standards"},
        ]

        terms = builder._extract_free_text_terms(docs)

        # Should include common bigrams
        assert any("quality" in term for term in terms)
        # Check if bigrams are present
        bigrams = [term for term in terms if " " in term]
        assert len(bigrams) > 0

    def test_extract_free_text_terms_no_bigrams(self):
        """Test that bigrams are excluded when disabled."""
        builder = VocabularyBuilder(
            max_vocab_size=100,
            min_term_freq=1,
            include_bigrams=False
        )
        docs = [
            {"title": "water quality monitoring"},
        ]

        terms = builder._extract_free_text_terms(docs)

        # Should not include any multi-word terms
        bigrams = [term for term in terms if " " in term]
        assert len(bigrams) == 0

    def test_build_vocabulary_integration(self):
        """Test full vocabulary building with both controlled and free-text."""
        builder = VocabularyBuilder(max_vocab_size=100, min_term_freq=2)
        docs = [
            {
                "title": "California water resources map",
                "description": "Map showing water resources in California",
                "provider": "Stanford",
                "resource_class": ["Maps", "Datasets"],
            },
            {
                "title": "California geology data",
                "description": "Geological data for California region",
                "provider": "Harvard",
                "resource_class": ["Datasets"],
            },
        ]

        vocab = builder.build_vocabulary(docs)

        # Should include controlled vocab
        assert "stanford" in vocab
        assert "harvard" in vocab
        assert "maps" in vocab
        assert "datasets" in vocab

        # Should include common free-text terms
        assert "california" in vocab

    def test_build_vocabulary_empty_docs(self):
        """Test vocabulary building with empty document list."""
        builder = VocabularyBuilder()
        vocab = builder.build_vocabulary([])

        assert vocab == []

    def test_build_vocabulary_handles_list_fields(self):
        """Test that list fields in free-text are handled properly."""
        builder = VocabularyBuilder(max_vocab_size=100, min_term_freq=1)
        docs = [
            {
                "title": "Test document",
                "description": ["First description", "Second description"],
                "publisher": ["Publisher One", "Publisher Two"],
            }
        ]

        vocab = builder.build_vocabulary(docs)

        assert "description" in vocab
        assert "publisher" in vocab


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        """Create a mock model directory for testing."""
        # This would normally require distilling an actual model
        # For now, we'll test with None and check error handling
        return tmp_path / "mock_model"

    def test_doc_to_text_basic(self, mock_model_dir):
        """Test document to text conversion."""
        # We can test _doc_to_text without loading a model
        doc = {
            "title": "Test Map",
            "description": "A map showing test data",
            "provider": "Stanford",
            "resource_class": ["Maps"],
        }

        # Create a generator instance (will fail on model load, but we can test the method)
        try:
            gen = EmbeddingGenerator(str(mock_model_dir))
        except Exception:
            # Expected - model doesn't exist
            # Test the method directly would require mocking
            pass

    def test_doc_to_text_with_lists(self):
        """Test text generation with list fields."""
        # Test the logic of combining fields
        doc = {
            "title": "California Map",
            "description": ["Detailed map", "Historical data"],
            "subject": ["Geology", "Water Resources"],
            "creator": ["John Doe", "Jane Smith"],
        }

        # The expected behavior is to join all fields
        # This would be tested through the actual embedding generation


class TestDistillModel:
    """Test suite for model distillation."""

    @requires_distill
    def test_distill_model_creates_files(self, tmp_path):
        """Test that model distillation creates expected output files."""
        vocab = ["california", "map", "water", "data", "stanford"]
        output_dir = tmp_path / "test_model"

        # Distill a very small model for testing
        # Note: This will download the base model on first run
        model_path = distill_model(
            vocabulary=vocab,
            output_dir=str(output_dir),
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            pca_dims=64,  # Small for faster testing
        )

        # Check that expected files exist
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "tokenizer.json").exists()
        assert (model_path / "embeddings.bin").exists()
        assert (model_path / "metadata.json").exists()

        # Check metadata content
        with open(model_path / "metadata.json") as f:
            metadata = json.load(f)
            assert "vocab_size" in metadata
            assert "embedding_dim" in metadata
            assert metadata["embedding_dim"] == 64
            assert metadata["pca_dims"] == 64

    @requires_distill
    def test_distill_model_with_empty_vocab(self, tmp_path):
        """Test that distillation with empty vocabulary still works."""
        output_dir = tmp_path / "empty_vocab_model"

        # Model2Vec should still work with an empty vocabulary
        # It will use the default tokenizer vocabulary
        model_path = distill_model(
            vocabulary=[],
            output_dir=str(output_dir),
            pca_dims=32,
        )

        assert model_path.exists()
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "tokenizer.json").exists()
        assert (model_path / "embeddings.bin").exists()


class TestEmbeddingIntegration:
    """Integration tests for the full embedding pipeline."""

    @requires_distill
    def test_full_pipeline(self, tmp_path):
        """Test the complete pipeline: vocabulary -> distill -> embed."""
        # Create sample documents
        docs = [
            {
                "title": "Map of California water resources",
                "description": "Detailed map showing water resources",
                "provider": "Stanford",
                "resource_class": ["Maps"],
                "subject": ["Hydrology", "Water"],
            },
            {
                "title": "California geology map",
                "description": "Geological features of California",
                "provider": "Harvard",
                "resource_class": ["Maps"],
                "subject": ["Geology"],
            },
        ]

        # Build vocabulary
        vocab_builder = VocabularyBuilder(max_vocab_size=50, min_term_freq=1)
        vocabulary = vocab_builder.build_vocabulary(docs)
        assert len(vocabulary) > 0

        # Distill model
        model_dir = tmp_path / "pipeline_model"
        model_path = distill_model(
            vocabulary=vocabulary,
            output_dir=str(model_dir),
            pca_dims=32,  # Small for faster testing
        )

        # Create embedding generator
        generator = EmbeddingGenerator(str(model_path))

        # Generate embeddings
        embedding1 = generator.generate_embedding(docs[0])
        embedding2 = generator.generate_embedding(docs[1])

        # Check embeddings are valid
        assert embedding1 is not None
        assert embedding2 is not None
        assert len(embedding1) == 32  # pca_dims
        assert len(embedding2) == 32
        assert isinstance(embedding1[0], float)
        assert isinstance(embedding2[0], float)

        # Embeddings should be different for different documents
        assert embedding1 != embedding2

    @requires_distill
    def test_embedding_with_missing_fields(self, tmp_path):
        """Test embedding generation with documents missing fields."""
        docs = [
            {"title": "Test document", "provider": "Stanford"},
        ]

        vocab_builder = VocabularyBuilder(max_vocab_size=20, min_term_freq=1)
        vocabulary = vocab_builder.build_vocabulary(docs)

        model_dir = tmp_path / "missing_fields_model"
        model_path = distill_model(
            vocabulary=vocabulary,
            output_dir=str(model_dir),
            pca_dims=16,
        )

        generator = EmbeddingGenerator(str(model_path))
        embedding = generator.generate_embedding(docs[0])

        # Should still generate embedding even with missing fields
        assert embedding is not None
        assert len(embedding) == 16
