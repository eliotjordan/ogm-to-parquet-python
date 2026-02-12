"""Tests for ogm_to_parquet.converters module."""

import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ogm_to_parquet.converters import (
    convert_to_flatgeobuf,
    convert_to_pmtiles,
    convert_to_pyramidal_tiff,
    extract_archive,
    is_image_format,
    is_vector_format,
)


class TestConvertToFlatgeobuf:
    """Tests for convert_to_flatgeobuf function."""

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_successful_conversion(self, mock_run: MagicMock, tmp_path: Path):
        """Should return True on successful conversion."""
        mock_run.return_value = MagicMock(returncode=0)

        input_path = tmp_path / "input.geojson"
        input_path.write_text('{"type": "FeatureCollection"}')
        output_path = tmp_path / "output.fgb"

        result = convert_to_flatgeobuf(input_path, output_path)

        assert result is True
        mock_run.assert_called_once()
        # Verify ogr2ogr command structure
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "ogr2ogr"
        assert "-f" in call_args[0][0]
        assert "FlatGeobuf" in call_args[0][0]

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_failed_conversion(self, mock_run: MagicMock, tmp_path: Path):
        """Should return False when ogr2ogr fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error: invalid input")

        input_path = tmp_path / "input.geojson"
        input_path.write_text('{"type": "FeatureCollection"}')
        output_path = tmp_path / "output.fgb"

        result = convert_to_flatgeobuf(input_path, output_path)

        assert result is False

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_timeout_handling(self, mock_run: MagicMock, tmp_path: Path):
        """Should return False on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("ogr2ogr", 600)

        input_path = tmp_path / "input.geojson"
        input_path.write_text('{"type": "FeatureCollection"}')
        output_path = tmp_path / "output.fgb"

        result = convert_to_flatgeobuf(input_path, output_path, timeout=600)

        assert result is False

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_sets_partial_reprojection_env(self, mock_run: MagicMock, tmp_path: Path):
        """Should set OGR_ENABLE_PARTIAL_REPROJECTION environment variable."""
        mock_run.return_value = MagicMock(returncode=0)

        input_path = tmp_path / "input.geojson"
        input_path.write_text('{"type": "FeatureCollection"}')
        output_path = tmp_path / "output.fgb"

        convert_to_flatgeobuf(input_path, output_path)

        call_args = mock_run.call_args
        env = call_args[1]["env"]
        assert env.get("OGR_ENABLE_PARTIAL_REPROJECTION") == "YES"


class TestConvertToPmtiles:
    """Tests for convert_to_pmtiles function."""

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_successful_conversion_with_zg(self, mock_run: MagicMock, tmp_path: Path):
        """Should return True on successful conversion with -zg flag."""
        mock_run.return_value = MagicMock(returncode=0)

        input_path = tmp_path / "input.fgb"
        input_path.write_bytes(b"dummy data")
        output_path = tmp_path / "output" / "dataset.pmtiles"

        result = convert_to_pmtiles(input_path, output_path)

        assert result is True
        # Verify output directory was created
        assert output_path.parent.exists()
        # Verify tippecanoe command with -zg
        call_args = mock_run.call_args
        assert "-zg" in call_args[0][0]

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_fallback_without_zg(self, mock_run: MagicMock, tmp_path: Path):
        """Should retry without -zg flag when first attempt fails."""
        # First call fails, second succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="Error with -zg"),
            MagicMock(returncode=0),
        ]

        input_path = tmp_path / "input.fgb"
        input_path.write_bytes(b"dummy data")
        output_path = tmp_path / "output.pmtiles"

        result = convert_to_pmtiles(input_path, output_path)

        assert result is True
        assert mock_run.call_count == 2
        # Second call should not have -zg
        second_call = mock_run.call_args_list[1]
        assert "-zg" not in second_call[0][0]

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_both_attempts_fail(self, mock_run: MagicMock, tmp_path: Path):
        """Should return False when both attempts fail."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        input_path = tmp_path / "input.fgb"
        input_path.write_bytes(b"dummy data")
        output_path = tmp_path / "output.pmtiles"

        result = convert_to_pmtiles(input_path, output_path)

        assert result is False

    @patch("ogm_to_parquet.converters.subprocess.run")
    def test_timeout_handling(self, mock_run: MagicMock, tmp_path: Path):
        """Should return False on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("tippecanoe", 1800)

        input_path = tmp_path / "input.fgb"
        input_path.write_bytes(b"dummy data")
        output_path = tmp_path / "output.pmtiles"

        result = convert_to_pmtiles(input_path, output_path, timeout=1800)

        assert result is False


class TestConvertToPyramidalTiff:
    """Tests for convert_to_pyramidal_tiff function."""

    @patch("ogm_to_parquet.converters.pyvips")
    def test_successful_conversion(self, mock_pyvips: MagicMock, tmp_path: Path):
        """Should return True on successful conversion."""
        mock_image = MagicMock()
        mock_pyvips.Image.new_from_file.return_value = mock_image

        input_path = tmp_path / "input.jpg"
        input_path.write_bytes(b"dummy image data")
        output_path = tmp_path / "output" / "dataset.tif"

        result = convert_to_pyramidal_tiff(input_path, output_path)

        assert result is True
        # Verify output directory was created
        assert output_path.parent.exists()
        # Verify tiffsave was called with correct parameters
        mock_image.tiffsave.assert_called_once()
        call_args = mock_image.tiffsave.call_args
        assert call_args[1]["compression"] == "jpeg"
        assert call_args[1]["pyramid"] is True
        assert call_args[1]["tile"] is True

    @patch("ogm_to_parquet.converters.pyvips")
    def test_custom_quality_and_tile_size(self, mock_pyvips: MagicMock, tmp_path: Path):
        """Should use custom quality and tile size."""
        mock_image = MagicMock()
        mock_pyvips.Image.new_from_file.return_value = mock_image

        input_path = tmp_path / "input.jpg"
        input_path.write_bytes(b"dummy data")
        output_path = tmp_path / "output.tif"

        convert_to_pyramidal_tiff(input_path, output_path, quality=85, tile_size=512)

        call_args = mock_image.tiffsave.call_args
        assert call_args[1]["Q"] == 85
        assert call_args[1]["tile_width"] == 512
        assert call_args[1]["tile_height"] == 512

    @patch("ogm_to_parquet.converters.pyvips")
    def test_vips_error(self, mock_pyvips: MagicMock, tmp_path: Path):
        """Should return False on pyvips error."""
        # Create a proper exception class for pyvips.Error
        mock_pyvips.Error = type("Error", (Exception,), {})
        mock_pyvips.Image.new_from_file.side_effect = mock_pyvips.Error("Invalid image")

        input_path = tmp_path / "input.jpg"
        input_path.write_bytes(b"invalid data")
        output_path = tmp_path / "output.tif"

        result = convert_to_pyramidal_tiff(input_path, output_path)

        assert result is False


class TestExtractArchive:
    """Tests for extract_archive function."""

    def test_extract_shapefile(self, tmp_path: Path):
        """Should extract and return path to .shp file."""
        # Create a zip with shapefile
        archive_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("data/file.shp", "shapefile data")
            zf.writestr("data/file.dbf", "dbf data")
            zf.writestr("data/file.shx", "shx data")

        result = extract_archive(archive_path, extract_dir)

        assert result is not None
        assert result.suffix == ".shp"
        assert result.exists()

    def test_extract_geojson(self, tmp_path: Path):
        """Should extract and return path to .geojson file."""
        archive_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("data.geojson", '{"type": "FeatureCollection"}')

        result = extract_archive(archive_path, extract_dir)

        assert result is not None
        assert result.suffix == ".geojson"

    def test_extract_nested_zip(self, tmp_path: Path):
        """Should handle nested zip files."""
        archive_path = tmp_path / "outer.zip"
        extract_dir = tmp_path / "extracted"

        # Create inner zip
        inner_zip = tmp_path / "inner.zip"
        with zipfile.ZipFile(inner_zip, "w") as zf:
            zf.writestr("data.geojson", '{"type": "FeatureCollection"}')

        # Create outer zip containing inner
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(inner_zip, "data.zip")

        result = extract_archive(archive_path, extract_dir)

        # Should return path to unzipped-data directory
        assert result is not None
        assert result.name == "unzipped-data"

    def test_invalid_zip(self, tmp_path: Path):
        """Should return None for invalid zip."""
        archive_path = tmp_path / "invalid.zip"
        archive_path.write_bytes(b"not a zip file")
        extract_dir = tmp_path / "extracted"

        result = extract_archive(archive_path, extract_dir)

        assert result is None

    def test_no_recognized_files(self, tmp_path: Path):
        """Should return None if no recognized data files found."""
        archive_path = tmp_path / "data.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("readme.txt", "Just a readme")
            zf.writestr("data.csv", "col1,col2")

        result = extract_archive(archive_path, extract_dir)

        assert result is None


class TestIsVectorFormat:
    """Tests for is_vector_format function."""

    def test_vector_formats(self):
        """Should return True for vector formats."""
        assert is_vector_format("GeoJSON") is True
        assert is_vector_format("Shapefile") is True
        assert is_vector_format("GeoPackage") is True
        assert is_vector_format("KML") is True

    def test_image_formats(self):
        """Should return False for image formats."""
        assert is_vector_format("JPEG") is False
        assert is_vector_format("TIFF") is False

    def test_unknown_format(self):
        """Should return False for unknown formats."""
        assert is_vector_format("Unknown") is False


class TestIsImageFormat:
    """Tests for is_image_format function."""

    def test_image_formats(self):
        """Should return True for image formats."""
        assert is_image_format("JPEG") is True
        assert is_image_format("TIFF") is True
        assert is_image_format("JPEG2000") is True

    def test_vector_formats(self):
        """Should return False for vector formats."""
        assert is_image_format("GeoJSON") is False
        assert is_image_format("Shapefile") is False

    def test_unknown_format(self):
        """Should return False for unknown formats."""
        assert is_image_format("Unknown") is False
