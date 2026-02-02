"""Tests for harvest module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ogm_to_parquet.harvest import FIELD_MAP, OgmToParquet


class TestFieldMapping:
    """Test suite for field mapping."""

    def test_field_map_keys(self):
        """Test that FIELD_MAP contains expected GeoBlacklight keys."""
        assert "dct_title_s" in FIELD_MAP
        assert "schema_provider_s" in FIELD_MAP
        assert "gbl_resourceClass_sm" in FIELD_MAP

    def test_field_map_values(self):
        """Test that FIELD_MAP maps to simplified names."""
        assert FIELD_MAP["dct_title_s"] == "title"
        assert FIELD_MAP["schema_provider_s"] == "provider"
        assert FIELD_MAP["gbl_resourceClass_sm"] == "resource_class"


class TestOgmToParquet:
    """Test suite for OgmToParquet class."""

    @pytest.fixture
    def harvester(self, tmp_path):
        """Create a harvester instance with temporary paths."""
        ogm_path = tmp_path / "opengeometadata"
        output_path = tmp_path / "output.parquet"
        ogm_path.mkdir()
        return OgmToParquet(str(ogm_path), str(output_path))

    def test_initialization(self, tmp_path):
        """Test harvester initialization."""
        ogm_path = str(tmp_path / "ogm")
        output_path = str(tmp_path / "output.parquet")
        harvester = OgmToParquet(ogm_path, output_path)

        assert harvester.ogm_path == Path(ogm_path)
        assert harvester.output_path == Path(output_path)
        assert harvester.rows == []

    def test_remap_doc_keys(self, harvester):
        """Test document key remapping."""
        doc = {
            "dct_title_s": "Test Title",
            "schema_provider_s": "Test Provider",
            "unmapped_field": "Keep As Is",
        }

        result = harvester._remap_doc_keys(doc)

        assert result["title"] == "Test Title"
        assert result["provider"] == "Test Provider"
        assert result["unmapped_field"] == "Keep As Is"
        assert "dct_title_s" not in result

    def test_clean_values_string(self, harvester):
        """Test cleaning single quotes from string values."""
        doc = {"title": "Test's Title with 'quotes'", "provider": "Test Provider"}

        result = harvester._clean_values(doc)

        assert result["title"] == "Tests Title with quotes"
        assert result["provider"] == "Test Provider"

    def test_clean_values_array(self, harvester):
        """Test cleaning single quotes from array values."""
        doc = {
            "creators": ["John's Name", "Jane Doe", "Bob's Lab"],
            "subjects": ["Test"],
        }

        result = harvester._clean_values(doc)

        assert result["creators"] == ["Johns Name", "Jane Doe", "Bobs Lab"]
        assert result["subjects"] == ["Test"]

    def test_clean_array(self, harvester):
        """Test array cleaning with mixed types."""
        arr = ["Text with 'quotes'", 123, "Another's text", None]

        result = harvester._clean_array(arr)

        assert result[0] == "Text with quotes"
        assert result[1] == 123
        assert result[2] == "Anothers text"
        assert result[3] is None

    def test_ensure_list_with_none(self, harvester):
        """Test ensure_list with None value."""
        assert harvester._ensure_list(None) is None

    def test_ensure_list_with_list(self, harvester):
        """Test ensure_list with list value."""
        value = ["a", "b", "c"]
        assert harvester._ensure_list(value) == ["a", "b", "c"]

    def test_ensure_list_with_empty_list(self, harvester):
        """Test ensure_list with empty list returns None."""
        assert harvester._ensure_list([]) is None

    def test_ensure_list_with_scalar(self, harvester):
        """Test ensure_list with scalar value."""
        assert harvester._ensure_list("single") == ["single"]
        assert harvester._ensure_list(123) == [123]

    def test_ensure_string_with_none(self, harvester):
        """Test ensure_string with None value."""
        assert harvester._ensure_string(None) is None

    def test_ensure_string_with_string(self, harvester):
        """Test ensure_string with string value."""
        assert harvester._ensure_string("test") == "test"

    def test_ensure_string_with_list(self, harvester):
        """Test ensure_string with list value."""
        assert harvester._ensure_string(["a", "b", "c"]) == "a b c"
        assert harvester._ensure_string(["single"]) == "single"

    def test_ensure_string_with_empty_list(self, harvester):
        """Test ensure_string with empty list returns None."""
        assert harvester._ensure_string([]) is None

    def test_ensure_string_with_list_containing_none(self, harvester):
        """Test ensure_string filters None from lists."""
        assert harvester._ensure_string(["a", None, "b"]) == "a b"

    def test_ensure_string_with_number(self, harvester):
        """Test ensure_string converts numbers to strings."""
        assert harvester._ensure_string(123) == "123"

    def test_ensure_float_list_with_none(self, harvester):
        """Test ensure_float_list with None value."""
        assert harvester._ensure_float_list(None) is None

    def test_ensure_float_list_with_int_list(self, harvester):
        """Test ensure_float_list with list of integers."""
        result = harvester._ensure_float_list([1869, 1870, 1871])
        assert result == [1869.0, 1870.0, 1871.0]

    def test_ensure_float_list_with_string_list(self, harvester):
        """Test ensure_float_list with list of string numbers."""
        result = harvester._ensure_float_list(["1869", "1870", "1871"])
        assert result == [1869.0, 1870.0, 1871.0]

    def test_ensure_float_list_with_mixed_list(self, harvester):
        """Test ensure_float_list with mixed types."""
        result = harvester._ensure_float_list([1869, "1870", 1871.5])
        assert result == [1869.0, 1870.0, 1871.5]

    def test_ensure_float_list_with_empty_list(self, harvester):
        """Test ensure_float_list with empty list returns None."""
        assert harvester._ensure_float_list([]) is None

    def test_ensure_float_list_with_scalar(self, harvester):
        """Test ensure_float_list converts scalar to list."""
        result = harvester._ensure_float_list(1869)
        assert result == [1869.0]

    def test_ensure_float_list_with_invalid_strings(self, harvester):
        """Test ensure_float_list skips invalid values."""
        result = harvester._ensure_float_list(["1869", "invalid", "1870"])
        assert result == [1869.0, 1870.0]

    def test_ensure_float_list_with_none_in_list(self, harvester):
        """Test ensure_float_list filters None from list."""
        result = harvester._ensure_float_list([1869, None, 1870])
        assert result == [1869.0, 1870.0]

    def test_extract_geojson_with_bbox(self, harvester):
        """Test GeoJSON extraction from bbox field."""
        doc = {"bbox": "ENVELOPE(-122, -121, 38, 37)"}

        result = harvester._extract_geojson(doc)

        assert result is not None
        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"

    def test_extract_geojson_without_bbox(self, harvester):
        """Test GeoJSON extraction when no bbox field."""
        doc = {"title": "Test"}

        result = harvester._extract_geojson(doc)

        assert result is None

    def test_extract_thumbnail_url_schema_org(self, harvester):
        """Test thumbnail extraction with schema.org URL."""
        references = {
            "http://schema.org/thumbnailUrl": "https://example.com/thumb.jpg",
            "http://iiif.io/api/image": "https://example.com/iiif/info.json",
        }
        doc = {"references": json.dumps(references)}

        result = harvester._extract_thumbnail_url(doc)

        assert result == "https://example.com/thumb.jpg"

    def test_extract_thumbnail_url_iiif(self, harvester):
        """Test thumbnail extraction with IIIF URL."""
        references = {"http://iiif.io/api/image": "https://example.com/iiif/info.json"}
        doc = {"references": json.dumps(references)}

        result = harvester._extract_thumbnail_url(doc)

        assert result == "https://example.com/iiif/square/150,150/0/default.jpg"

    def test_extract_thumbnail_url_no_references(self, harvester):
        """Test thumbnail extraction with no references field."""
        doc = {"title": "Test"}

        result = harvester._extract_thumbnail_url(doc)

        assert result is None

    def test_extract_thumbnail_url_empty_references(self, harvester):
        """Test thumbnail extraction with empty references."""
        doc = {"references": ""}

        result = harvester._extract_thumbnail_url(doc)

        assert result is None

    def test_extract_thumbnail_url_invalid_json(self, harvester):
        """Test thumbnail extraction with invalid JSON references."""
        doc = {"references": "not valid json"}

        result = harvester._extract_thumbnail_url(doc)

        assert result is None

    def test_build_row(self, harvester):
        """Test building a row for Parquet output."""
        doc = {
            "id": "test-123",
            "title": "Test Title",
            "creator": ["John Doe"],
            "provider": "Test Provider",
            "resource_class": ["Maps"],
            "bbox": "ENVELOPE(-122, -121, 38, 37)",
            "references": json.dumps(
                {"http://schema.org/thumbnailUrl": "https://example.com/thumb.jpg"}
            ),
        }

        row = harvester._build_row(doc)

        assert row["id"] == "test-123"
        assert row["title"] == "Test Title"
        assert row["creator"] == ["John Doe"]
        assert row["provider"] == "Test Provider"
        assert row["resource_class"] == ["Maps"]
        assert row["thumbnail"] == "https://example.com/thumb.jpg"
        assert row["geojson"] is not None
        # Verify embeddings field exists (may be None if embeddings disabled)
        assert "embeddings" in row

    def test_collect_documents_empty_directory(self, harvester):
        """Test collecting documents from empty directory."""
        docs = harvester._collect_documents()

        assert docs == []

    def test_collect_documents_with_json_files(self, harvester, tmp_path):
        """Test collecting documents from directory with JSON files."""
        ogm_path = tmp_path / "opengeometadata"
        ogm_path.mkdir(exist_ok=True)

        # Create test JSON files
        doc1 = {"id": "doc1", "dct_title_s": "Title 1"}
        doc2 = {"id": "doc2", "dct_title_s": "Title 2"}

        (ogm_path / "doc1.json").write_text(json.dumps(doc1))

        subdir = ogm_path / "subdir"
        subdir.mkdir()
        (subdir / "doc2.json").write_text(json.dumps(doc2))

        harvester = OgmToParquet(str(ogm_path), str(tmp_path / "output.parquet"))
        docs = harvester._collect_documents()

        assert len(docs) == 2
        ids = {doc["id"] for doc in docs}
        assert ids == {"doc1", "doc2"}

    def test_collect_documents_invalid_json(self, harvester, tmp_path, caplog):
        """Test collecting documents handles invalid JSON gracefully."""
        ogm_path = tmp_path / "opengeometadata"
        ogm_path.mkdir(exist_ok=True)

        # Create invalid JSON file
        (ogm_path / "invalid.json").write_text("{ invalid json")

        # Create valid JSON file
        (ogm_path / "valid.json").write_text(json.dumps({"id": "valid"}))

        harvester = OgmToParquet(str(ogm_path), str(tmp_path / "output.parquet"))
        docs = harvester._collect_documents()

        # Should collect only valid document
        assert len(docs) == 1
        assert docs[0]["id"] == "valid"

    def test_convert_with_sample_data(self, tmp_path):
        """Test full conversion process with sample data."""
        ogm_path = tmp_path / "opengeometadata"
        output_path = tmp_path / "output.parquet"
        ogm_path.mkdir()

        # Create sample document
        doc = {
            "id": "test-123",
            "dct_title_s": "Test Title",
            "schema_provider_s": "Test Provider",
            "gbl_resourceClass_sm": ["Maps"],
            "dcat_bbox": "ENVELOPE(-122, -121, 38, 37)",
        }
        (ogm_path / "test.json").write_text(json.dumps(doc))

        harvester = OgmToParquet(str(ogm_path), str(output_path))
        harvester.convert()

        assert len(harvester.rows) == 1
        assert harvester.rows[0]["id"] == "test-123"
        assert harvester.rows[0]["title"] == "Test Title"
        assert output_path.exists()

    def test_remap_and_clean_integration(self, harvester):
        """Test integration of remapping and cleaning."""
        doc = {
            "dct_title_s": "Test's Title",
            "schema_provider_s": "Provider's Name",
            "dct_creator_sm": ["John's Lab", "Jane's Team"],
        }

        result = harvester._remap_and_clean(doc)

        assert result["title"] == "Tests Title"
        assert result["provider"] == "Providers Name"
        assert result["creator"] == ["Johns Lab", "Janes Team"]

    def test_write_parquet_no_rows(self, harvester, caplog):
        """Test writing Parquet with no rows logs warning."""
        harvester._write_parquet()

        assert "No rows to write" in caplog.text

    def test_convert_handles_errors_gracefully(self, tmp_path, caplog):
        """Test that convert handles document processing errors gracefully."""
        ogm_path = tmp_path / "opengeometadata"
        output_path = tmp_path / "output.parquet"
        ogm_path.mkdir()

        # Create document that will cause error (missing 'id')
        bad_doc = {"dct_title_s": "No ID Document"}
        (ogm_path / "bad.json").write_text(json.dumps(bad_doc))

        # Create valid document
        good_doc = {"id": "good-123", "dct_title_s": "Good Document"}
        (ogm_path / "good.json").write_text(json.dumps(good_doc))

        harvester = OgmToParquet(str(ogm_path), str(output_path))
        harvester.convert()

        # Should process valid document despite error with bad document
        assert len(harvester.rows) >= 1
        assert output_path.exists()

    def test_collect_documents_skips_non_dict_json(self, tmp_path):
        """Test that non-dict JSON documents are skipped."""
        ogm_path = tmp_path / "opengeometadata"
        output_path = tmp_path / "output.parquet"
        ogm_path.mkdir()

        # Create JSON file with string instead of dict
        (ogm_path / "string.json").write_text(json.dumps("just a string"))

        # Create JSON file with array
        (ogm_path / "array.json").write_text(json.dumps([1, 2, 3]))

        # Create valid dict document
        (ogm_path / "valid.json").write_text(json.dumps({"id": "valid-123"}))

        harvester = OgmToParquet(str(ogm_path), str(output_path))
        docs = harvester._collect_documents()

        # Should only collect the valid dict document
        assert len(docs) == 1
        assert docs[0]["id"] == "valid-123"

    def test_convert_skips_non_dict_documents(self, tmp_path, caplog):
        """Test that convert skips non-dict documents gracefully."""
        ogm_path = tmp_path / "opengeometadata"
        output_path = tmp_path / "output.parquet"
        ogm_path.mkdir()

        # Create non-dict JSON
        (ogm_path / "string.json").write_text(json.dumps("string document"))

        # Create valid document
        valid_doc = {"id": "valid-123", "dct_title_s": "Valid Document"}
        (ogm_path / "valid.json").write_text(json.dumps(valid_doc))

        harvester = OgmToParquet(str(ogm_path), str(output_path))
        harvester.convert()

        # Should process only the valid document
        assert len(harvester.rows) == 1
        assert harvester.rows[0]["id"] == "valid-123"
        assert output_path.exists()
