"""Tests for the derivatives processing module."""

import sqlite3
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from ogm_to_parquet.config import IMAGE_FORMATS, VECTOR_FORMATS
from ogm_to_parquet.converters import extract_archive
from ogm_to_parquet.derivative_jobs import (
    increment_retry_count,
    process_derivative,
    update_status,
)
from ogm_to_parquet.derivatives import DerivativeProcessor
from ogm_to_parquet.utils import (
    clean_doc_id,
    cleanup_scratch,
    download_file,
    generate_output_path,
)


class TestGenerateOutputPath:
    """Tests for output path generation."""

    def test_standard_id(self, tmp_path):
        """Test path generation with standard 32-char ID."""
        output = generate_output_path(tmp_path, "060c078970f84019a0c0426016c9d151", ".pmtiles")
        expected = (
            tmp_path / "06" / "0c" / "07" / "060c078970f84019a0c0426016c9d151" / "dataset.pmtiles"
        )
        assert output == expected

    def test_tiff_extension(self, tmp_path):
        """Test path generation with TIFF extension."""
        output = generate_output_path(tmp_path, "abc123def456", ".tif")
        assert output.name == "dataset.tif"

    def test_short_id_padded(self, tmp_path):
        """Test that short IDs are zero-padded."""
        output = generate_output_path(tmp_path, "abc", ".pmtiles")
        # "abc" padded to "000abc"
        expected = tmp_path / "00" / "0a" / "bc" / "000abc" / "dataset.pmtiles"
        assert output == expected

    def test_nested_directory_structure(self, tmp_path):
        """Test that first 6 chars create nested directories."""
        output = generate_output_path(tmp_path, "aabbccdd", ".pmtiles")
        assert output.parts[-5:-1] == ("aa", "bb", "cc", "aabbccdd")

    def test_strips_ark_prefix(self, tmp_path):
        """Test that ark-NNNNN- prefix is stripped."""
        output = generate_output_path(tmp_path, "ark-85335-s1rn4xzk", ".pmtiles")
        assert output.parts[-2] == "s1rn4xzk"

    def test_strips_rutgers_prefix(self, tmp_path):
        """Test that rutgers-lib: prefix is stripped."""
        output = generate_output_path(tmp_path, "rutgers-lib:24776", ".pmtiles")
        assert output.parts[-2] == "024776"  # Padded because "24776" is 5 chars

    def test_strips_stanford_prefix(self, tmp_path):
        """Test that stanford- prefix is stripped."""
        output = generate_output_path(tmp_path, "stanford-pn863fv0810", ".pmtiles")
        assert output.parts[-2] == "pn863fv0810"

    def test_no_prefix_unchanged(self, tmp_path):
        """Test that IDs without recognized prefixes are unchanged."""
        output = generate_output_path(tmp_path, "harvard-abc123", ".pmtiles")
        assert output.parts[-2] == "harvard-abc123"


class TestCleanDocId:
    """Tests for document ID prefix cleaning."""

    def test_clean_ark_prefix(self):
        """Test cleaning ark-NNNNN- prefix."""
        assert clean_doc_id("ark-85335-s1rn4xzk") == "s1rn4xzk"

    def test_clean_rutgers_prefix(self):
        """Test cleaning rutgers-lib: prefix."""
        assert clean_doc_id("rutgers-lib:24776") == "24776"

    def test_clean_stanford_prefix(self):
        """Test cleaning stanford- prefix."""
        assert clean_doc_id("stanford-pn863fv0810") == "pn863fv0810"

    def test_no_prefix_unchanged(self):
        """Test that IDs without recognized prefixes are unchanged."""
        assert (
            clean_doc_id("060c078970f84019a0c0426016c9d151") == "060c078970f84019a0c0426016c9d151"
        )

    def test_partial_match_unchanged(self):
        """Test that partial matches are not stripped."""
        assert clean_doc_id("ark-text-123") == "ark-text-123"  # ark- without digits
        assert clean_doc_id("stanford") == "stanford"  # No trailing dash


class TestDownloadFile:
    """Tests for file downloading."""

    def test_successful_download(self, tmp_path):
        """Test successful file download."""
        dest = tmp_path / "test.geojson"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'{"type": "FeatureCollection"}']

        with patch("ogm_to_parquet.utils.requests.get") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_response)
            mock_get.return_value = mock_response
            result = download_file("http://example.com/data.geojson", dest)

        assert result is True
        assert dest.exists()

    def test_download_creates_parent_dirs(self, tmp_path):
        """Test that download creates parent directories."""
        dest = tmp_path / "nested" / "path" / "test.json"

        with patch("ogm_to_parquet.utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response
            download_file("http://example.com/data.json", dest)

        assert dest.parent.exists()

    def test_download_failure_returns_false(self, tmp_path):
        """Test that download failure returns False."""
        import requests as req

        dest = tmp_path / "test.json"

        with patch("ogm_to_parquet.utils.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Network error")
            result = download_file("http://example.com/data.json", dest)

        assert result is False


class TestExtractArchive:
    """Tests for archive extraction."""

    def test_extract_shapefile(self, tmp_path):
        """Test extracting a shapefile from zip."""
        # Create a mock zip with shapefile
        zip_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/layer.shp", b"shapefile content")
            zf.writestr("data/layer.shx", b"index content")
            zf.writestr("data/layer.dbf", b"dbf content")

        result = extract_archive(zip_path, extract_dir)

        assert result is not None
        assert result.suffix == ".shp"

    def test_extract_geojson(self, tmp_path):
        """Test extracting a GeoJSON from zip."""
        zip_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.geojson", b'{"type": "FeatureCollection"}')

        result = extract_archive(zip_path, extract_dir)

        assert result is not None
        assert result.suffix == ".geojson"

    def test_extract_no_recognized_files(self, tmp_path):
        """Test extraction with no recognized files returns None."""
        zip_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", b"Hello")

        result = extract_archive(zip_path, extract_dir)

        assert result is None

    def test_extract_invalid_zip(self, tmp_path):
        """Test that invalid zip returns None."""
        invalid_zip = tmp_path / "bad.zip"
        invalid_zip.write_bytes(b"not a zip file")
        extract_dir = tmp_path / "extracted"

        result = extract_archive(invalid_zip, extract_dir)

        assert result is None


class TestDatabaseOperations:
    """Tests for database operations."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a test database."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE cloud_derivatives (
                id TEXT PRIMARY KEY,
                download_url TEXT,
                format TEXT,
                derivative_url TEXT,
                status TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO cloud_derivatives (id, download_url, format, status)
            VALUES ('test123', 'http://example.com/data.zip', 'Shapefile', 'unprocessed')
            """
        )
        conn.commit()
        conn.close()
        return str(db_file)

    def test_update_status_complete(self, db_path):
        """Test updating status to complete with derivative URL."""
        update_status(db_path, "test123", "complete", derivative_url="/path/to/output.pmtiles")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, derivative_url FROM cloud_derivatives WHERE id = ?", ("test123",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "complete"
        assert row[1] == "/path/to/output.pmtiles"

    def test_update_status_error(self, db_path):
        """Test updating status to error with message."""
        update_status(db_path, "test123", "error", error_message="Something went wrong")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, error_message FROM cloud_derivatives WHERE id = ?", ("test123",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "error"
        assert row[1] == "Something went wrong"

    def test_increment_retry_count(self, db_path):
        """Test incrementing retry count."""
        # First increment
        count = increment_retry_count(db_path, "test123")
        assert count == 1

        # Second increment
        count = increment_retry_count(db_path, "test123")
        assert count == 2


class TestCleanupScratch:
    """Tests for scratch directory cleanup."""

    def test_cleanup_removes_directory(self, tmp_path):
        """Test that cleanup removes the scratch directory."""
        scratch_dir = tmp_path / "scratch"
        doc_dir = scratch_dir / "doc123"
        doc_dir.mkdir(parents=True)
        (doc_dir / "test.fgb").touch()

        cleanup_scratch(scratch_dir, "doc123")

        assert not doc_dir.exists()

    def test_cleanup_handles_missing_dir(self, tmp_path):
        """Test that cleanup handles missing directories gracefully."""
        scratch_dir = tmp_path / "scratch"
        # Should not raise
        cleanup_scratch(scratch_dir, "nonexistent")


class TestProcessDerivative:
    """Tests for the main processing function."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a test database."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE cloud_derivatives (
                id TEXT PRIMARY KEY,
                download_url TEXT,
                format TEXT,
                derivative_url TEXT,
                status TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO cloud_derivatives (id, download_url, format, status)
            VALUES ('test123', 'http://example.com/data.geojson', 'GeoJSON', 'unprocessed')
            """
        )
        conn.commit()
        conn.close()
        return str(db_file)

    def test_unsupported_format(self, tmp_path, db_path):
        """Test that unsupported formats return error."""
        result = process_derivative(
            doc_id="test123",
            download_url="http://example.com/data.xyz",
            doc_format="Unknown",
            db_path=db_path,
            scratch_dir=str(tmp_path / "scratch"),
            output_dir=str(tmp_path / "output"),
        )

        assert result["success"] is False
        assert "Unsupported format" in result["error"]

    def test_download_failure(self, tmp_path, db_path):
        """Test handling of download failure."""
        with patch("ogm_to_parquet.derivative_jobs.download_file", return_value=False):
            result = process_derivative(
                doc_id="test123",
                download_url="http://example.com/data.geojson",
                doc_format="GeoJSON",
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )

        assert result["success"] is False
        assert "Failed to download" in result["error"]

    def test_vector_processing_success(self, tmp_path, db_path):
        """Test successful vector file processing."""
        with (
            patch("ogm_to_parquet.derivative_jobs.download_file", return_value=True),
            patch("ogm_to_parquet.derivative_jobs.convert_to_flatgeobuf", return_value=True),
            patch("ogm_to_parquet.derivative_jobs.convert_to_pmtiles", return_value=True),
        ):
            result = process_derivative(
                doc_id="test123",
                download_url="http://example.com/data.geojson",
                doc_format="GeoJSON",
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )

        assert result["success"] is True
        assert "output_path" in result
        assert result["output_path"].endswith(".pmtiles")

    def test_image_processing_success(self, tmp_path, db_path):
        """Test successful image file processing."""
        # Update test record to image format
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE cloud_derivatives SET format = 'JPEG' WHERE id = 'test123'")
        conn.commit()
        conn.close()

        with (
            patch("ogm_to_parquet.derivative_jobs.download_file", return_value=True),
            patch("ogm_to_parquet.derivative_jobs.convert_to_pyramidal_tiff", return_value=True),
        ):
            result = process_derivative(
                doc_id="test123",
                download_url="http://example.com/image.jpg",
                doc_format="JPEG",
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )

        assert result["success"] is True
        assert "output_path" in result
        assert result["output_path"].endswith(".tif")

    def test_cleanup_on_error(self, tmp_path, db_path):
        """Test that scratch is cleaned up even on error."""
        scratch_dir = tmp_path / "scratch"

        with (
            patch("ogm_to_parquet.derivative_jobs.download_file", return_value=True),
            patch("ogm_to_parquet.derivative_jobs.convert_to_flatgeobuf", return_value=False),
        ):
            process_derivative(
                doc_id="test123",
                download_url="http://example.com/data.geojson",
                doc_format="GeoJSON",
                db_path=db_path,
                scratch_dir=str(scratch_dir),
                output_dir=str(tmp_path / "output"),
            )

        # Scratch directory for doc should be cleaned up
        assert not (scratch_dir / "test123").exists()


class TestDerivativeProcessor:
    """Tests for the DerivativeProcessor class."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a test database."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE cloud_derivatives (
                id TEXT PRIMARY KEY,
                download_url TEXT,
                format TEXT,
                derivative_url TEXT,
                status TEXT,
                error_message TEXT
            )
            """
        )
        # Insert test data
        cursor.executemany(
            """
            INSERT INTO cloud_derivatives (id, download_url, format, status)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("doc1", "http://example.com/a.geojson", "GeoJSON", "unprocessed"),
                ("doc2", "http://example.com/b.zip", "Shapefile", "unprocessed"),
                ("doc3", "http://example.com/c.tif", "TIFF", "in_progress"),
                ("doc4", "http://example.com/d.geojson", "GeoJSON", "complete"),
                ("doc5", "http://example.com/e.geojson", "GeoJSON", "error"),
            ],
        )
        conn.commit()
        conn.close()
        return str(db_file)

    def test_ensure_retry_column_adds_column(self, tmp_path, db_path):
        """Test that ensure_retry_column adds the column."""
        with patch("ogm_to_parquet.derivatives.Redis"):
            processor = DerivativeProcessor(
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )
            processor.ensure_retry_column()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(cloud_derivatives)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        assert "retry_count" in columns

    def test_ensure_retry_column_idempotent(self, tmp_path, db_path):
        """Test that ensure_retry_column can be called multiple times."""
        with patch("ogm_to_parquet.derivatives.Redis"):
            processor = DerivativeProcessor(
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )
            # Should not raise on multiple calls
            processor.ensure_retry_column()
            processor.ensure_retry_column()

    def test_get_pending_documents(self, tmp_path, db_path):
        """Test fetching pending documents."""
        with patch("ogm_to_parquet.derivatives.Redis"):
            processor = DerivativeProcessor(
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )
            docs = processor.get_pending_documents()

        # Should get only unprocessed, not in_progress, complete, or error
        doc_ids = {d["id"] for d in docs}
        assert "doc1" in doc_ids  # unprocessed
        assert "doc2" in doc_ids  # unprocessed
        assert "doc3" not in doc_ids  # in_progress
        assert "doc4" not in doc_ids  # complete
        assert "doc5" not in doc_ids  # error

    def test_get_specific_document(self, tmp_path, db_path):
        """Test fetching a specific document by ID."""
        with patch("ogm_to_parquet.derivatives.Redis"):
            processor = DerivativeProcessor(
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )
            docs = processor.get_pending_documents(doc_id="doc4")

        # Should get the specific doc regardless of status
        assert len(docs) == 1
        assert docs[0]["id"] == "doc4"

    def test_get_pending_returns_required_fields(self, tmp_path, db_path):
        """Test that returned documents have required fields."""
        with patch("ogm_to_parquet.derivatives.Redis"):
            processor = DerivativeProcessor(
                db_path=db_path,
                scratch_dir=str(tmp_path / "scratch"),
                output_dir=str(tmp_path / "output"),
            )
            docs = processor.get_pending_documents()

        for doc in docs:
            assert "id" in doc
            assert "download_url" in doc
            assert "format" in doc


class TestDryRun:
    """Tests for dry-run functionality."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a test database."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE cloud_derivatives (
                id TEXT PRIMARY KEY,
                download_url TEXT,
                format TEXT,
                derivative_url TEXT,
                status TEXT,
                error_message TEXT
            )
            """
        )
        cursor.executemany(
            """
            INSERT INTO cloud_derivatives (id, download_url, format, status)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("doc1", "http://example.com/a.geojson", "GeoJSON", "unprocessed"),
                ("doc2", "http://example.com/b.zip", "Shapefile", "unprocessed"),
                ("doc3", "http://example.com/c.tif", "TIFF", "unprocessed"),
            ],
        )
        conn.commit()
        conn.close()
        return str(db_file)

    def test_dry_run_no_redis_connection(self, tmp_path, db_path):
        """Test that dry-run works without connecting to Redis."""
        # This should not raise even with invalid Redis URL
        processor = DerivativeProcessor(
            db_path=db_path,
            scratch_dir=str(tmp_path / "scratch"),
            output_dir=str(tmp_path / "output"),
            redis_url="redis://nonexistent:6379",
        )

        # get_pending_documents should work without Redis
        docs = processor.get_pending_documents()
        assert len(docs) == 3

    def test_dry_run_groups_by_format(self, tmp_path, db_path):
        """Test that documents can be retrieved and grouped by format."""
        processor = DerivativeProcessor(
            db_path=db_path,
            scratch_dir=str(tmp_path / "scratch"),
            output_dir=str(tmp_path / "output"),
            redis_url="redis://localhost:6379",
        )

        docs = processor.get_pending_documents()

        # Group by format like dry-run does
        by_format: dict[str, list] = {}
        for doc in docs:
            fmt = doc["format"]
            if fmt not in by_format:
                by_format[fmt] = []
            by_format[fmt].append(doc)

        assert "GeoJSON" in by_format
        assert "Shapefile" in by_format
        assert "TIFF" in by_format
        assert len(by_format["GeoJSON"]) == 1
        assert len(by_format["Shapefile"]) == 1
        assert len(by_format["TIFF"]) == 1


class TestFormatConstants:
    """Tests for format constant definitions."""

    def test_vector_formats_defined(self):
        """Test that expected vector formats are defined."""
        assert "GeoJSON" in VECTOR_FORMATS
        assert "Shapefile" in VECTOR_FORMATS
        assert "GeoPackage" in VECTOR_FORMATS
        assert "KML" in VECTOR_FORMATS

    def test_image_formats_defined(self):
        """Test that expected image formats are defined."""
        assert "JPEG" in IMAGE_FORMATS
        assert "JPEG2000" in IMAGE_FORMATS
        assert "TIFF" in IMAGE_FORMATS

    def test_formats_disjoint(self):
        """Test that vector and image formats don't overlap."""
        assert len(VECTOR_FORMATS & IMAGE_FORMATS) == 0
