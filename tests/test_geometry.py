"""Tests for geometry module."""

import json

import pytest

from ogm_to_parquet.geometry import Geometry


class TestGeometry:
    """Test suite for Geometry class."""

    def test_simple_wkt_polygon_to_geojson(self):
        """Test conversion of simple WKT POLYGON to GeoJSON."""
        wkt = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"
        geom = Geometry(wkt)
        result = geom.to_geojson()

        # Parse and validate
        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"
        assert len(geojson_obj["coordinates"]) == 1
        assert len(geojson_obj["coordinates"][0]) == 5  # Closed ring

    def test_envelope_to_geojson(self):
        """Test conversion of ENVELOPE syntax to GeoJSON."""
        envelope = "ENVELOPE(-180, 180, 90, -90)"
        geom = Geometry(envelope)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"
        # ENVELOPE(minx, maxx, maxy, miny) -> POLYGON
        coords = geojson_obj["coordinates"][0]
        # Should form a rectangle
        assert len(coords) == 5  # Closed ring
        assert coords[0] == coords[-1]  # First and last point same

    def test_envelope_with_whitespace(self):
        """Test ENVELOPE parsing with various whitespace."""
        envelope = "  ENVELOPE( -90.71 , 30.67 , 29.20 , -89.35 )  "
        geom = Geometry(envelope)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"

    def test_invalid_geometry_returns_default_extent(self):
        """Test that invalid geometry returns world extent."""
        invalid_wkt = "INVALID GEOMETRY STRING"
        geom = Geometry(invalid_wkt)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"
        # World extent coordinates
        coords = geojson_obj["coordinates"][0]
        assert [-180.0, 90.0] in coords
        assert [180.0, -90.0] in coords

    def test_bounding_box_from_polygon(self):
        """Test bounding box extraction from WKT polygon."""
        wkt = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"
        geom = Geometry(wkt)
        bbox = geom.bounding_box()

        # Format: "w, s, e, n"
        parts = [float(x.strip()) for x in bbox.split(",")]
        assert len(parts) == 4
        minx, miny, maxx, maxy = parts
        assert minx == 10.0
        assert miny == 10.0
        assert maxx == 40.0
        assert maxy == 40.0

    def test_bounding_box_from_envelope(self):
        """Test bounding box extraction from ENVELOPE syntax."""
        envelope = "ENVELOPE(-90.71, -89.35, 30.67, 29.20)"
        geom = Geometry(envelope)
        bbox = geom.bounding_box()

        parts = [float(x.strip()) for x in bbox.split(",")]
        assert len(parts) == 4
        minx, miny, maxx, maxy = parts
        assert minx == -90.71
        assert maxx == -89.35
        assert miny == 29.20
        assert maxy == 30.67

    def test_bounding_box_invalid_geometry_returns_world(self):
        """Test that invalid geometry returns world extent for bbox."""
        invalid = "NOT A VALID GEOMETRY"
        geom = Geometry(invalid)
        bbox = geom.bounding_box()

        assert bbox == "-180.0, -90.0, 180.0, 90.0"

    def test_envelope_to_polygon_conversion(self):
        """Test internal ENVELOPE to POLYGON conversion."""
        envelope = "ENVELOPE(-90, 90, 45, -45)"
        geom = Geometry(envelope)
        wkt = geom._geometry_as_wkt()

        # Should be a valid WKT polygon
        assert wkt.startswith("POLYGON")
        assert "-90" in wkt
        assert "90" in wkt
        assert "45" in wkt
        assert "-45" in wkt

    def test_point_geometry(self):
        """Test conversion of WKT POINT to GeoJSON."""
        wkt = "POINT (30 10)"
        geom = Geometry(wkt)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Point"
        assert geojson_obj["coordinates"] == [30.0, 10.0]

    def test_linestring_geometry(self):
        """Test conversion of WKT LINESTRING to GeoJSON."""
        wkt = "LINESTRING (30 10, 10 30, 40 40)"
        geom = Geometry(wkt)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "LineString"
        assert len(geojson_obj["coordinates"]) == 3

    def test_multipolygon_geometry(self):
        """Test conversion of WKT MULTIPOLYGON to GeoJSON."""
        wkt = "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"
        geom = Geometry(wkt)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "MultiPolygon"
        assert len(geojson_obj["coordinates"]) == 2

    def test_case_insensitive_envelope(self):
        """Test that ENVELOPE parsing is case-insensitive."""
        envelope_lower = "envelope(-90, 90, 45, -45)"
        geom = Geometry(envelope_lower)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"

    def test_negative_coordinates(self):
        """Test handling of negative coordinates in ENVELOPE."""
        envelope = "ENVELOPE(-122.5, -122.0, 37.8, 37.7)"
        geom = Geometry(envelope)
        result = geom.to_geojson()

        geojson_obj = json.loads(result)
        assert geojson_obj["type"] == "Polygon"
        # Verify coordinates are negative
        coords = geojson_obj["coordinates"][0]
        assert any(c[0] < 0 for c in coords)  # Negative longitude

    def test_decimal_coordinates(self):
        """Test handling of decimal coordinates."""
        envelope = "ENVELOPE(-122.419, -122.365, 37.792, 37.754)"
        geom = Geometry(envelope)
        bbox = geom.bounding_box()

        parts = [float(x.strip()) for x in bbox.split(",")]
        assert all(isinstance(p, float) for p in parts)
        # Check precision is maintained
        assert -122.42 < parts[0] < -122.41
