"""Tests for text_extract module."""

import json
import sqlite3
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from ogm_to_parquet.text_extract import TextExtractor


class TestTextExtractorInit:
    """Test TextExtractor initialization."""

    def test_default_initialization(self, tmp_path):
        """Test default initialization values."""
        extractor = TextExtractor()
        assert extractor.db_path == Path("./tmp/enrichment.db")
        assert extractor.scratch_dir == Path("./tmp/scratch")
        assert extractor.ollama_url == "http://localhost:11434"
        assert extractor.model == "qwen3-vl:30b"
        assert extractor.max_retries == 3
        assert extractor.timeout == 300

    def test_custom_initialization(self, tmp_path):
        """Test custom initialization values."""
        extractor = TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(tmp_path / "scratch"),
            ollama_url="http://custom:8080",
            model="custom-model",
            max_retries=5,
            timeout=600,
        )
        assert extractor.db_path == tmp_path / "test.db"
        assert extractor.scratch_dir == tmp_path / "scratch"
        assert extractor.ollama_url == "http://custom:8080"
        assert extractor.model == "custom-model"
        assert extractor.max_retries == 5
        assert extractor.timeout == 600


class TestExtractJson:
    """Test JSON extraction from response content."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a TextExtractor instance."""
        return TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(tmp_path / "scratch"),
        )

    def test_valid_json_direct(self, extractor):
        """Test extracting valid JSON directly."""
        content = '{"settlements": ["Town A", "Town B"]}'
        result = extractor._extract_json(content)
        assert result == content

    def test_json_in_markdown_code_block(self, extractor):
        """Test extracting JSON from markdown code block."""
        content = '```json\n{"roads": ["Main St", "Oak Ave"]}\n```'
        result = extractor._extract_json(content)
        assert result == '{"roads": ["Main St", "Oak Ave"]}'

    def test_json_in_generic_code_block(self, extractor):
        """Test extracting JSON from generic code block."""
        content = '```\n{"buildings": ["Church", "Library"]}\n```'
        result = extractor._extract_json(content)
        assert result == '{"buildings": ["Church", "Library"]}'

    def test_json_embedded_in_text(self, extractor):
        """Test extracting JSON embedded in surrounding text."""
        content = 'Here is the result:\n{"regions": ["North", "South"]}\nEnd of output.'
        result = extractor._extract_json(content)
        assert result == '{"regions": ["North", "South"]}'

    def test_invalid_json_returns_none(self, extractor):
        """Test that invalid JSON returns None."""
        content = "This is not JSON at all"
        result = extractor._extract_json(content)
        assert result is None

    def test_malformed_json_returns_none(self, extractor):
        """Test that malformed JSON returns None."""
        content = '{"unclosed": "bracket"'
        result = extractor._extract_json(content)
        assert result is None

    def test_empty_json_object(self, extractor):
        """Test extracting empty JSON object."""
        content = "{}"
        result = extractor._extract_json(content)
        assert result == "{}"

    def test_complex_nested_json(self, extractor):
        """Test extracting complex nested JSON."""
        data = {
            "settlements": ["Town A", "Town B"],
            "roads": ["Main St", "Oak Ave"],
            "waterbodies": ["Lake X"],
        }
        content = json.dumps(data)
        result = extractor._extract_json(content)
        assert json.loads(result) == data


class TestPrepareImage:
    """Test image preparation functionality."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a TextExtractor instance."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(scratch),
        )

    def test_valid_small_jpeg(self, extractor, tmp_path):
        """Test that small JPEG is returned as-is."""
        # Create a small JPEG
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.jpg"
        img.save(img_path, "JPEG")

        result = extractor._prepare_image(img_path)
        assert result == img_path

    def test_invalid_extension_returns_none(self, extractor, tmp_path):
        """Test that invalid extension returns None."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not an image")

        result = extractor._prepare_image(txt_path)
        assert result is None

    def test_png_converted_to_jpeg(self, extractor, tmp_path):
        """Test that PNG is converted to JPEG."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "test.png"
        img.save(img_path, "PNG")

        result = extractor._prepare_image(img_path)
        assert result is not None
        assert result.suffix == ".jpg"

    def test_rgba_image_converted(self, extractor, tmp_path):
        """Test that RGBA image is converted to RGB JPEG."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img_path = tmp_path / "test.png"
        img.save(img_path, "PNG")

        result = extractor._prepare_image(img_path)
        assert result is not None
        assert result.suffix == ".jpg"

        # Verify the result is RGB
        with Image.open(result) as converted:
            assert converted.mode == "RGB"


class TestCleanupFile:
    """Test file cleanup functionality."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a TextExtractor instance."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(scratch),
        )

    def test_cleanup_single_file(self, extractor, tmp_path):
        """Test cleaning up a single file."""
        test_file = tmp_path / "test.jpg"
        test_file.write_text("test")

        extractor._cleanup_file(test_file)
        assert not test_file.exists()

    def test_cleanup_extracted_directory_when_empty(self, extractor, tmp_path):
        """Test cleaning up an extracted directory when it becomes empty."""
        extract_dir = tmp_path / "doc123_extracted"
        extract_dir.mkdir()
        test_file = extract_dir / "image.jpg"
        test_file.write_text("test")

        extractor._cleanup_file(test_file)
        assert not extract_dir.exists()

    def test_cleanup_keeps_directory_with_other_files(self, extractor, tmp_path):
        """Test that cleanup doesn't delete directory if other files remain."""
        extract_dir = tmp_path / "doc456_extracted"
        extract_dir.mkdir()
        original_file = extract_dir / "image.jp2"
        original_file.write_text("original")
        converted_file = extract_dir / "image.jpg"
        converted_file.write_text("converted")

        # Cleanup the original file - directory should remain because converted file exists
        extractor._cleanup_file(original_file)
        assert not original_file.exists()
        assert extract_dir.exists()
        assert converted_file.exists()

    def test_cleanup_nonexistent_file(self, extractor, tmp_path):
        """Test cleaning up a nonexistent file doesn't raise."""
        nonexistent = tmp_path / "nonexistent.jpg"
        # Should not raise
        extractor._cleanup_file(nonexistent)


class TestExtractZip:
    """Test zip extraction functionality."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a TextExtractor instance."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(scratch),
        )

    def test_extract_zip_with_image(self, extractor):
        """Test extracting a zip file containing an image."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color="green")
        img_bytes = b""
        import io

        buffer = io.BytesIO()
        img.save(buffer, "JPEG")
        img_bytes = buffer.getvalue()

        # Create zip file
        zip_path = extractor.scratch_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("image.jpg", img_bytes)

        result = extractor._extract_zip(zip_path, "test-doc")

        assert result is not None
        assert result.suffix == ".jpg"
        assert not zip_path.exists()  # Zip should be deleted

    def test_extract_zip_no_image(self, extractor):
        """Test extracting a zip file without images returns None."""
        zip_path = extractor.scratch_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "No images here")

        result = extractor._extract_zip(zip_path, "test-doc")

        assert result is None
        assert not zip_path.exists()


class TestMarkError:
    """Test error marking functionality."""

    @pytest.fixture
    def extractor_with_db(self, tmp_path):
        """Create a TextExtractor with initialized database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE text_extraction (
                id TEXT PRIMARY KEY,
                image_url TEXT,
                format TEXT,
                generated_output TEXT,
                status TEXT,
                error_message TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO text_extraction VALUES (?, ?, ?, ?, ?, ?)",
            ("test-doc", "http://example.com/img.jpg", "TIFF", "", "in_progress", None),
        )
        conn.commit()
        conn.close()

        return TextExtractor(
            db_path=str(db_path),
            scratch_dir=str(tmp_path / "scratch"),
        )

    def test_mark_error_updates_status(self, extractor_with_db):
        """Test that mark_error updates the document status."""
        conn = sqlite3.connect(extractor_with_db.db_path)
        extractor_with_db._mark_error(conn, "test-doc")

        cursor = conn.cursor()
        cursor.execute("SELECT status, error_message FROM text_extraction WHERE id = ?", ("test-doc",))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "error"
        assert row[1] is None  # No error message provided

    def test_mark_error_with_message(self, extractor_with_db):
        """Test that mark_error saves the error message."""
        conn = sqlite3.connect(extractor_with_db.db_path)
        extractor_with_db._mark_error(conn, "test-doc", "Download failed: 404 Not Found")

        cursor = conn.cursor()
        cursor.execute("SELECT status, error_message FROM text_extraction WHERE id = ?", ("test-doc",))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "error"
        assert row[1] == "Download failed: 404 Not Found"


class TestCheckOllamaConnection:
    """Test Ollama connection checking."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a TextExtractor instance."""
        return TextExtractor(
            db_path=str(tmp_path / "test.db"),
            scratch_dir=str(tmp_path / "scratch"),
            model="test-model:latest",
        )

    def test_connection_success_with_model(self, extractor):
        """Test successful connection when model is available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "test-model:latest"}, {"name": "other-model:v1"}]
        }

        with patch("requests.get", return_value=mock_response):
            result = extractor.check_ollama_connection()

        assert result is True

    def test_connection_success_model_partial_match(self, extractor):
        """Test successful connection with partial model name match."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "test-model:v2"}, {"name": "other-model:v1"}]
        }

        with patch("requests.get", return_value=mock_response):
            result = extractor.check_ollama_connection()

        assert result is True

    def test_connection_model_not_found(self, extractor):
        """Test connection when model is not available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "other-model:v1"}]}

        with patch("requests.get", return_value=mock_response):
            result = extractor.check_ollama_connection()

        assert result is False

    def test_connection_error(self, extractor):
        """Test handling of connection error."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            result = extractor.check_ollama_connection()

        assert result is False


class TestProcessAllIntegration:
    """Integration tests for process_all (with mocked external calls)."""

    @pytest.fixture
    def extractor_with_docs(self, tmp_path):
        """Create a TextExtractor with test documents in database."""
        db_path = tmp_path / "test.db"
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE text_extraction (
                id TEXT PRIMARY KEY,
                image_url TEXT,
                format TEXT,
                generated_output TEXT,
                status TEXT,
                error_message TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO text_extraction VALUES (?, ?, ?, ?, ?, ?)",
            ("doc1", "http://example.com/img1.jpg", "TIFF", "", "unprocessed", None),
        )
        cursor.execute(
            "INSERT INTO text_extraction VALUES (?, ?, ?, ?, ?, ?)",
            ("doc2", "", "JPEG", "", "unprocessed", None),
        )
        conn.commit()
        conn.close()

        return TextExtractor(
            db_path=str(db_path),
            scratch_dir=str(scratch),
        )

    def test_process_all_skips_empty_url(self, extractor_with_docs):
        """Test that documents with empty URLs are skipped."""
        with (
            patch.object(extractor_with_docs, "check_ollama_connection", return_value=True),
            patch.object(extractor_with_docs, "_process_document") as mock_process,
        ):
            mock_process.return_value = "completed"
            results = extractor_with_docs.process_all()

        # doc2 should be skipped due to empty URL
        assert results["skipped"] == 1

    def test_process_all_aborts_if_ollama_unavailable(self, extractor_with_docs):
        """Test that processing aborts if Ollama connection fails."""
        with patch.object(extractor_with_docs, "check_ollama_connection", return_value=False):
            results = extractor_with_docs.process_all()

        # Should return empty stats without processing anything
        assert results == {"completed": 0, "errors": 0, "skipped": 0}

    def test_process_single_document_by_id(self, extractor_with_docs):
        """Test processing a single document by ID."""
        with (
            patch.object(extractor_with_docs, "check_ollama_connection", return_value=True),
            patch.object(extractor_with_docs, "_process_document") as mock_process,
        ):
            mock_process.return_value = "completed"
            results = extractor_with_docs.process_all(doc_id="doc1")

        # Should only process doc1
        assert mock_process.call_count == 1
        call_args = mock_process.call_args[0]
        assert call_args[1] == "doc1"  # doc_id argument
        assert results["completed"] == 1

    def test_process_single_document_not_found(self, extractor_with_docs):
        """Test processing a document ID that doesn't exist."""
        with patch.object(extractor_with_docs, "check_ollama_connection", return_value=True):
            results = extractor_with_docs.process_all(doc_id="nonexistent-doc")

        # Should return empty stats since document not found
        assert results == {"completed": 0, "errors": 0, "skipped": 0}

    def test_process_single_document_ignores_status(self, extractor_with_docs):
        """Test that processing by ID works regardless of document status."""
        # Update doc1 to have 'complete' status
        conn = sqlite3.connect(extractor_with_docs.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE text_extraction SET status = 'complete' WHERE id = 'doc1'")
        conn.commit()
        conn.close()

        with (
            patch.object(extractor_with_docs, "check_ollama_connection", return_value=True),
            patch.object(extractor_with_docs, "_process_document") as mock_process,
        ):
            mock_process.return_value = "completed"
            results = extractor_with_docs.process_all(doc_id="doc1")

        # Should still process doc1 even though it was already 'complete'
        assert mock_process.call_count == 1
        assert results["completed"] == 1
