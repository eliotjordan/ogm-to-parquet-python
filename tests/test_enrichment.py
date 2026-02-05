"""Tests for enrichment module."""

import json
import sqlite3

import pytest

from ogm_to_parquet.enrichment import (
    DERIVATIVE_FORMATS,
    DOWNLOAD_URL_KEY,
    EXCLUDED_SERVICE_KEYS,
    IIIF_IMAGE_KEY,
    EnrichmentPreparer,
)


class TestEnrichmentPreparer:
    """Test suite for EnrichmentPreparer class."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer with temporary database paths."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def test_initialization(self, tmp_path):
        """Test preparer initialization with custom paths."""
        derivatives_db = tmp_path / "custom_derivatives.db"
        text_extraction_db = tmp_path / "custom_text.db"
        preparer = EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )
        assert preparer.derivatives_db_path == derivatives_db
        assert preparer.text_extraction_db_path == text_extraction_db

    def test_setup_derivatives_database_creates_table(self, preparer):
        """Test that setup_derivatives_database creates the required table."""
        result = preparer.setup_derivatives_database()
        assert result is True
        assert preparer.derivatives_db_path.exists()

        conn = sqlite3.connect(preparer.derivatives_db_path)
        cursor = conn.cursor()

        # Check cloud_derivatives table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cloud_derivatives'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_setup_text_extraction_database_creates_table(self, preparer):
        """Test that setup_text_extraction_database creates the required table."""
        result = preparer.setup_text_extraction_database()
        assert result is True
        assert preparer.text_extraction_db_path.exists()

        conn = sqlite3.connect(preparer.text_extraction_db_path)
        cursor = conn.cursor()

        # Check text_extraction table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='text_extraction'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_setup_database_skips_existing(self, preparer):
        """Test that setup methods return False for existing databases."""
        preparer.setup_derivatives_database()
        result = preparer.setup_derivatives_database()
        assert result is False

        preparer.setup_text_extraction_database()
        result = preparer.setup_text_extraction_database()
        assert result is False

    def test_cloud_derivatives_table_schema(self, preparer):
        """Test cloud_derivatives table has correct columns."""
        preparer.setup_derivatives_database()
        conn = sqlite3.connect(preparer.derivatives_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(cloud_derivatives)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "download_url" in columns
        assert "format" in columns
        assert "derivative_url" in columns
        assert "status" in columns
        assert "error_message" in columns

    def test_text_extraction_table_schema(self, preparer):
        """Test text_extraction table has correct columns."""
        preparer.setup_text_extraction_database()
        conn = sqlite3.connect(preparer.text_extraction_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(text_extraction)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "image_url" in columns
        assert "format" in columns
        assert "generated_output" in columns
        assert "status" in columns
        assert "error_message" in columns


class TestCloudDerivativesFiltering:
    """Test filtering logic for cloud_derivatives table."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer with temporary database paths."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def _make_doc(
        self,
        doc_id: str,
        access_rights: str = "Public",
        doc_format: str = "Shapefile",
        references: dict | None = None,
    ) -> dict:
        """Create a test document with specified attributes."""
        if references is None:
            references = {DOWNLOAD_URL_KEY: "https://example.com/download.zip"}
        return {
            "id": doc_id,
            "dct_accessRights_s": access_rights,
            "dct_format_s": doc_format,
            "dct_references_s": json.dumps(references),
        }

    def test_eligible_document_inserted_with_download_url_and_format(self, preparer):
        """Test that eligible documents are inserted with download_url and format."""
        doc = self._make_doc("test-123")
        results = preparer.prepare([doc])

        assert results["cloud_derivatives"] == 1

        conn = sqlite3.connect(preparer.derivatives_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, download_url, format, status FROM cloud_derivatives WHERE id = ?",
            ("test-123",),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-123"
        assert row[1] == "https://example.com/download.zip"
        assert row[2] == "Shapefile"
        assert row[3] == "unprocessed"

    def test_download_url_array_format_extracts_first(self, preparer):
        """Test that array format download URL extracts the first URL."""
        references = {
            DOWNLOAD_URL_KEY: [
                {"url": "https://example.com/first.zip", "label": "Shapefile"},
                {"url": "https://example.com/second.zip", "label": "GeoJSON"},
            ]
        }
        doc = self._make_doc("test-array", references=references)
        results = preparer.prepare([doc])

        assert results["cloud_derivatives"] == 1

        conn = sqlite3.connect(preparer.derivatives_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT download_url FROM cloud_derivatives WHERE id = ?", ("test-array",))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "https://example.com/first.zip"

    def test_non_public_access_excluded(self, preparer):
        """Test that documents with non-Public access are excluded."""
        doc = self._make_doc("test-123", access_rights="Restricted")
        results = preparer.prepare([doc])
        assert results["cloud_derivatives"] == 0

    def test_invalid_format_excluded(self, preparer):
        """Test that documents with invalid formats are excluded."""
        doc = self._make_doc("test-123", doc_format="PDF")
        results = preparer.prepare([doc])
        assert results["cloud_derivatives"] == 0

    def test_all_derivative_formats_accepted(self, preparer):
        """Test that all DERIVATIVE_FORMATS are accepted."""
        docs = [self._make_doc(f"test-{fmt}", doc_format=fmt) for fmt in DERIVATIVE_FORMATS]
        results = preparer.prepare(docs)
        assert results["cloud_derivatives"] == len(DERIVATIVE_FORMATS)

    def test_excluded_service_key_prevents_insertion(self, preparer):
        """Test that documents with excluded service keys are not inserted."""
        for excluded_key in list(EXCLUDED_SERVICE_KEYS)[:3]:  # Test a few
            references = {
                DOWNLOAD_URL_KEY: "https://example.com/download.zip",
                excluded_key: "https://example.com/service",
            }
            doc = self._make_doc(f"test-{hash(excluded_key)}", references=references)
            results = preparer.prepare([doc])
            assert results["cloud_derivatives"] == 0

    def test_missing_download_url_excluded(self, preparer):
        """Test that documents without download URL are excluded."""
        references = {"http://schema.org/url": "https://example.com/page"}
        doc = self._make_doc("test-123", references=references)
        results = preparer.prepare([doc])
        assert results["cloud_derivatives"] == 0

    def test_duplicate_id_skipped(self, preparer):
        """Test that duplicate IDs are skipped on subsequent runs."""
        doc = self._make_doc("test-123")

        results1 = preparer.prepare([doc])
        assert results1["cloud_derivatives"] == 1

        results2 = preparer.prepare([doc])
        assert results2["cloud_derivatives"] == 0

    def test_invalid_references_json_excluded(self, preparer):
        """Test that documents with invalid references JSON are excluded."""
        doc = {
            "id": "test-123",
            "dct_accessRights_s": "Public",
            "dct_format_s": "Shapefile",
            "dct_references_s": "not valid json",
        }
        results = preparer.prepare([doc])
        assert results["cloud_derivatives"] == 0


class TestTextExtractionFiltering:
    """Test filtering logic for text_extraction table."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer with temporary database paths."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def _make_doc(
        self,
        doc_id: str,
        access_rights: str = "Public",
        doc_format: str = "TIFF",
        references: dict | None = None,
    ) -> dict:
        """Create a test document with specified attributes."""
        if references is None:
            references = {IIIF_IMAGE_KEY: "https://example.com/iiif/info.json"}
        return {
            "id": doc_id,
            "dct_accessRights_s": access_rights,
            "dct_format_s": doc_format,
            "dct_references_s": json.dumps(references),
        }

    def test_iiif_document_inserted_with_image_url_and_format(self, preparer):
        """Test that documents with IIIF image reference are inserted with image_url and format."""
        doc = self._make_doc("test-123")
        results = preparer.prepare([doc])

        assert results["text_extraction"] == 1

        conn = sqlite3.connect(preparer.text_extraction_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, image_url, format, status FROM text_extraction WHERE id = ?", ("test-123",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-123"
        assert row[1] == "https://example.com/iiif/full/full/0/default.jpg"
        assert row[2] == "IIIF"
        assert row[3] == "unprocessed"

    def test_image_with_download_url_inserted_with_image_url_and_format(self, preparer):
        """Test that image documents with download URL are inserted with image_url and format."""
        references = {DOWNLOAD_URL_KEY: "https://example.com/image.tiff"}
        doc = self._make_doc("test-TIFF", doc_format="TIFF", references=references)
        results = preparer.prepare([doc])

        assert results["text_extraction"] == 1

        conn = sqlite3.connect(preparer.text_extraction_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, image_url, format FROM text_extraction WHERE id = ?", ("test-TIFF",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[1] == "https://example.com/image.tiff"
        assert row[2] == "TIFF"

    def test_non_public_access_excluded(self, preparer):
        """Test that documents with non-Public access are excluded."""
        doc = self._make_doc("test-123", access_rights="Restricted")
        results = preparer.prepare([doc])
        assert results["text_extraction"] == 0

    def test_non_image_format_without_iiif_excluded(self, preparer):
        """Test that non-image formats without IIIF are excluded."""
        references = {DOWNLOAD_URL_KEY: "https://example.com/data.zip"}
        doc = self._make_doc("test-123", doc_format="Shapefile", references=references)
        results = preparer.prepare([doc])
        assert results["text_extraction"] == 0

    def test_image_format_without_download_or_iiif_excluded(self, preparer):
        """Test that image formats without download URL or IIIF are excluded."""
        references = {"http://schema.org/url": "https://example.com/page"}
        doc = self._make_doc("test-123", doc_format="TIFF", references=references)
        results = preparer.prepare([doc])
        assert results["text_extraction"] == 0

    def test_duplicate_id_skipped(self, preparer):
        """Test that duplicate IDs are skipped on subsequent runs."""
        doc = self._make_doc("test-123")

        results1 = preparer.prepare([doc])
        assert results1["text_extraction"] == 1

        results2 = preparer.prepare([doc])
        assert results2["text_extraction"] == 0

    def test_both_iiif_and_download_only_inserts_once(self, preparer):
        """Test that document with both IIIF and download URL is only inserted once."""
        references = {
            IIIF_IMAGE_KEY: "https://example.com/iiif/info.json",
            DOWNLOAD_URL_KEY: "https://example.com/image.tiff",
        }
        doc = self._make_doc("test-123", doc_format="TIFF", references=references)
        results = preparer.prepare([doc])
        assert results["text_extraction"] == 1


class TestParseReferences:
    """Test the _parse_references helper method."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def test_parse_string_json(self, preparer):
        """Test parsing JSON string."""
        refs = '{"key": "value"}'
        result = preparer._parse_references(refs)
        assert result == {"key": "value"}

    def test_parse_dict_passthrough(self, preparer):
        """Test that dict is passed through."""
        refs = {"key": "value"}
        result = preparer._parse_references(refs)
        assert result == {"key": "value"}

    def test_parse_invalid_json_returns_none(self, preparer):
        """Test that invalid JSON returns None."""
        result = preparer._parse_references("not json")
        assert result is None

    def test_parse_none_returns_none(self, preparer):
        """Test that None input returns None."""
        result = preparer._parse_references(None)
        assert result is None

    def test_parse_empty_string_returns_none(self, preparer):
        """Test that empty string returns None."""
        result = preparer._parse_references("")
        assert result is None


class TestPrepareIntegration:
    """Integration tests for the prepare method."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def test_prepare_multiple_documents(self, preparer):
        """Test preparing multiple documents at once."""
        docs = [
            # Eligible for cloud_derivatives only
            {
                "id": "geo-1",
                "dct_accessRights_s": "Public",
                "dct_format_s": "Shapefile",
                "dct_references_s": json.dumps({DOWNLOAD_URL_KEY: "https://example.com/1.zip"}),
            },
            # Eligible for text_extraction only (IIIF)
            {
                "id": "img-1",
                "dct_accessRights_s": "Public",
                "dct_format_s": "PDF",
                "dct_references_s": json.dumps({IIIF_IMAGE_KEY: "https://example.com/iiif"}),
            },
            # Eligible for both
            {
                "id": "both-1",
                "dct_accessRights_s": "Public",
                "dct_format_s": "TIFF",
                "dct_references_s": json.dumps(
                    {
                        DOWNLOAD_URL_KEY: "https://example.com/image.tiff",
                        IIIF_IMAGE_KEY: "https://example.com/iiif",
                    }
                ),
            },
            # Eligible for neither (restricted)
            {
                "id": "restricted-1",
                "dct_accessRights_s": "Restricted",
                "dct_format_s": "Shapefile",
                "dct_references_s": json.dumps({DOWNLOAD_URL_KEY: "https://example.com/2.zip"}),
            },
        ]

        results = preparer.prepare(docs)

        # Only geo-1 is eligible for cloud_derivatives
        # (both-1 has IIIF which is in EXCLUDED_SERVICE_KEYS, so it's excluded)
        assert results["cloud_derivatives"] == 1
        # img-1 and both-1 are eligible for text_extraction
        assert results["text_extraction"] == 2

    def test_prepare_creates_databases_if_not_exist(self, preparer):
        """Test that prepare creates databases if they don't exist."""
        assert not preparer.derivatives_db_path.exists()
        assert not preparer.text_extraction_db_path.exists()
        preparer.prepare([])
        assert preparer.derivatives_db_path.exists()
        assert preparer.text_extraction_db_path.exists()


class TestGenerateImageUrl:
    """Test the _generate_image_url helper method."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def test_iiif_url_with_info_json(self, preparer):
        """Test IIIF URL transformation when ending with info.json."""
        refs = {IIIF_IMAGE_KEY: "https://example.com/iiif/image/info.json"}
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/iiif/image/full/full/0/default.jpg"

    def test_iiif_url_without_info_json(self, preparer):
        """Test IIIF URL transformation when not ending with info.json."""
        refs = {IIIF_IMAGE_KEY: "https://example.com/iiif/image"}
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/iiif/image/full/full/0/default.jpg"

    def test_iiif_url_with_trailing_slash(self, preparer):
        """Test IIIF URL transformation with trailing slash."""
        refs = {IIIF_IMAGE_KEY: "https://example.com/iiif/image/"}
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/iiif/image/full/full/0/default.jpg"

    def test_download_url_used_when_no_iiif(self, preparer):
        """Test that download URL is used when no IIIF reference."""
        refs = {DOWNLOAD_URL_KEY: "https://example.com/image.jpg"}
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/image.jpg"

    def test_iiif_preferred_over_download_url(self, preparer):
        """Test that IIIF is preferred over download URL."""
        refs = {
            IIIF_IMAGE_KEY: "https://example.com/iiif/info.json",
            DOWNLOAD_URL_KEY: "https://example.com/image.jpg",
        }
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/iiif/full/full/0/default.jpg"

    def test_returns_none_when_no_suitable_reference(self, preparer):
        """Test that None is returned when no suitable reference."""
        refs = {"http://schema.org/url": "https://example.com/page"}
        result = preparer._generate_image_url(refs)
        assert result is None

    def test_returns_none_for_empty_references(self, preparer):
        """Test that None is returned for empty references."""
        result = preparer._generate_image_url({})
        assert result is None

    def test_download_url_as_array_of_objects(self, preparer):
        """Test download URL extraction from array of objects."""
        refs = {
            DOWNLOAD_URL_KEY: [
                {"url": "https://example.com/first.jpg", "label": "Full"},
                {"url": "https://example.com/second.jpg", "label": "Thumb"},
            ]
        }
        result = preparer._generate_image_url(refs)
        assert result == "https://example.com/first.jpg"


class TestExtractDownloadUrl:
    """Test the _extract_download_url helper method."""

    @pytest.fixture
    def preparer(self, tmp_path):
        """Create an EnrichmentPreparer."""
        derivatives_db = tmp_path / "derivatives.db"
        text_extraction_db = tmp_path / "text_extraction.db"
        return EnrichmentPreparer(
            derivatives_db_path=str(derivatives_db),
            text_extraction_db_path=str(text_extraction_db),
        )

    def test_string_value(self, preparer):
        """Test extraction from string value."""
        result = preparer._extract_download_url("https://example.com/file.zip")
        assert result == "https://example.com/file.zip"

    def test_array_of_objects_first_url(self, preparer):
        """Test extraction from array of objects returns first URL."""
        value = [
            {"url": "https://example.com/first.zip", "label": "First"},
            {"url": "https://example.com/second.zip", "label": "Second"},
        ]
        result = preparer._extract_download_url(value)
        assert result == "https://example.com/first.zip"

    def test_array_with_single_object(self, preparer):
        """Test extraction from array with single object."""
        value = [{"url": "https://example.com/only.zip"}]
        result = preparer._extract_download_url(value)
        assert result == "https://example.com/only.zip"

    def test_empty_array_returns_none(self, preparer):
        """Test that empty array returns None."""
        result = preparer._extract_download_url([])
        assert result is None

    def test_array_without_url_key_returns_none(self, preparer):
        """Test that array object without 'url' key returns None."""
        value = [{"href": "https://example.com/file.zip"}]
        result = preparer._extract_download_url(value)
        assert result is None

    def test_none_value_returns_none(self, preparer):
        """Test that None value returns None."""
        result = preparer._extract_download_url(None)
        assert result is None

    def test_invalid_type_returns_none(self, preparer):
        """Test that invalid type returns None."""
        result = preparer._extract_download_url(12345)
        assert result is None
