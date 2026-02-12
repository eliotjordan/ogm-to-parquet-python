"""Tests for ogm_to_parquet.utils module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from ogm_to_parquet.utils import (
    check_ollama,
    check_prerequisites,
    check_redis,
    check_tool_available,
    clean_doc_id,
    cleanup_scratch,
    download_file,
    ensure_directory,
    generate_output_path,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Should return a logger instance."""
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_default_name(self):
        """Should use default name when not specified."""
        logger = setup_logging()
        assert logger.name == "ogm_to_parquet"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_directory(self, tmp_path: Path):
        """Should create directory if it doesn't exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()

        result = ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_existing_directory(self, tmp_path: Path):
        """Should not raise for existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_directory(existing_dir)

        assert result == existing_dir


class TestDownloadFile:
    """Tests for download_file function."""

    @patch("ogm_to_parquet.utils.requests.get")
    def test_successful_download(self, mock_get: MagicMock, tmp_path: Path):
        """Should download and save file."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_response

        dest = tmp_path / "downloaded.txt"
        result = download_file("http://example.com/file.txt", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"test content"

    @patch("ogm_to_parquet.utils.requests.get")
    def test_failed_download(self, mock_get: MagicMock, tmp_path: Path):
        """Should return False on download failure."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        dest = tmp_path / "failed.txt"
        result = download_file("http://example.com/file.txt", dest)

        assert result is False
        assert not dest.exists()

    @patch("ogm_to_parquet.utils.requests.get")
    def test_creates_parent_directory(self, mock_get: MagicMock, tmp_path: Path):
        """Should create parent directories if needed."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        dest = tmp_path / "nested" / "dir" / "file.txt"
        result = download_file("http://example.com/file.txt", dest)

        assert result is True
        assert dest.parent.exists()


class TestCheckRedis:
    """Tests for check_redis function."""

    def test_redis_available(self):
        """Should return True when Redis is available."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis.ping.return_value = True
            mock_redis_class.from_url.return_value = mock_redis

            result = check_redis("redis://localhost:6379")

            assert result is True
            mock_redis.ping.assert_called_once()

    def test_redis_unavailable(self):
        """Should return False when Redis connection fails."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = Exception("Connection refused")

            result = check_redis("redis://localhost:6379")

            assert result is False


class TestCheckToolAvailable:
    """Tests for check_tool_available function."""

    @patch("ogm_to_parquet.utils.subprocess.run")
    def test_tool_available(self, mock_run: MagicMock):
        """Should return True when tool is available."""
        mock_run.return_value = MagicMock()

        result = check_tool_available("ls")

        assert result is True

    @patch("ogm_to_parquet.utils.subprocess.run")
    def test_tool_not_found(self, mock_run: MagicMock):
        """Should return False when tool is not found."""
        mock_run.side_effect = FileNotFoundError()

        result = check_tool_available("nonexistent_tool")

        assert result is False


class TestCheckOllama:
    """Tests for check_ollama function."""

    @patch("ogm_to_parquet.utils.requests.get")
    def test_ollama_available(self, mock_get: MagicMock):
        """Should return True when Ollama is running."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = check_ollama("http://localhost:11434")

        assert result is True

    @patch("ogm_to_parquet.utils.requests.get")
    def test_ollama_unavailable(self, mock_get: MagicMock):
        """Should return False when Ollama is not running."""
        mock_get.side_effect = Exception("Connection refused")

        result = check_ollama("http://localhost:11434")

        assert result is False


class TestCheckPrerequisites:
    """Tests for check_prerequisites function."""

    @patch("ogm_to_parquet.utils.check_redis")
    @patch("ogm_to_parquet.utils.check_tool_available")
    def test_multiple_checks(self, mock_tool: MagicMock, mock_redis: MagicMock):
        """Should check multiple prerequisites."""
        mock_redis.return_value = True
        mock_tool.return_value = True

        result = check_prerequisites(["redis", "ogr2ogr", "tippecanoe"])

        assert result == {"redis": True, "ogr2ogr": True, "tippecanoe": True}

    def test_unknown_check(self):
        """Should return False for unknown check types."""
        result = check_prerequisites(["unknown_tool"])

        assert result == {"unknown_tool": False}


class TestCleanDocId:
    """Tests for clean_doc_id function."""

    def test_ark_prefix(self):
        """Should strip ark-NNNNN- prefix."""
        assert clean_doc_id("ark-85335-abc123") == "abc123"
        assert clean_doc_id("ark-12345-xyz789") == "xyz789"

    def test_rutgers_prefix(self):
        """Should strip rutgers-lib: prefix."""
        assert clean_doc_id("rutgers-lib:abc123") == "abc123"

    def test_stanford_prefix(self):
        """Should strip stanford- prefix."""
        assert clean_doc_id("stanford-abc123") == "abc123"

    def test_no_prefix(self):
        """Should return unchanged ID without known prefix."""
        assert clean_doc_id("abc123def456") == "abc123def456"

    def test_empty_string(self):
        """Should handle empty string."""
        assert clean_doc_id("") == ""


class TestGenerateOutputPath:
    """Tests for generate_output_path function."""

    def test_standard_id(self, tmp_path: Path):
        """Should generate nested path for standard ID."""
        result = generate_output_path(tmp_path, "060c078970f84019a0c0426016c9d151", ".pmtiles")

        expected = (
            tmp_path
            / "06"
            / "0c"
            / "07"
            / "060c078970f84019a0c0426016c9d151"
            / "dataset.pmtiles"
        )
        assert result == expected

    def test_strips_prefix(self, tmp_path: Path):
        """Should strip known prefixes before generating path."""
        result = generate_output_path(tmp_path, "stanford-abc123def456", ".tif")

        # After stripping 'stanford-', ID becomes 'abc123def456'
        expected = tmp_path / "ab" / "c1" / "23" / "abc123def456" / "dataset.tif"
        assert result == expected

    def test_short_id_padded(self, tmp_path: Path):
        """Should pad short IDs with zeros."""
        result = generate_output_path(tmp_path, "abc", ".pmtiles")

        # 'abc' -> '000abc' after padding
        expected = tmp_path / "00" / "0a" / "bc" / "000abc" / "dataset.pmtiles"
        assert result == expected


class TestCleanupScratch:
    """Tests for cleanup_scratch function."""

    def test_removes_directory(self, tmp_path: Path):
        """Should remove scratch directory for document."""
        doc_dir = tmp_path / "doc123"
        doc_dir.mkdir()
        (doc_dir / "file.txt").write_text("test")

        cleanup_scratch(tmp_path, "doc123")

        assert not doc_dir.exists()

    def test_nonexistent_directory(self, tmp_path: Path):
        """Should handle non-existent directory gracefully."""
        # Should not raise
        cleanup_scratch(tmp_path, "nonexistent")
