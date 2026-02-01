"""Geometry transformation utilities for OpenGeoMetadata.

Transforms and parses geometry expressed in WKT or CSW WKT ENVELOPE syntax.
Imported from GeoBlacklight:
https://github.com/geoblacklight/geoblacklight/blob/main/lib/geoblacklight/geometry.rb
"""

import json
import logging
import re
from typing import Optional

import geojson
from shapely import wkt
from shapely.geometry import box, mapping, shape

logger = logging.getLogger(__name__)


class Geometry:
    """Handles geometry transformations between WKT, ENVELOPE, and GeoJSON formats."""

    def __init__(self, geom: str):
        """Initialize with a WKT or ENVELOPE formatted string.

        Args:
            geom: WKT or WKT ENVELOPE syntax formatted string
        """
        self.geom = geom

    def to_geojson(self) -> str:
        """Convert geometry to GeoJSON string.

        Returns:
            JSON string representation of the geometry, or default world extent on error
        """
        try:
            wkt_string = self._geometry_as_wkt()
            geometry = wkt.loads(wkt_string)
            geojson_dict = mapping(geometry)
            return json.dumps(geojson_dict)
        except Exception as e:
            logger.warning(f"Geometry is not valid: {self.geom} - {e}")
            return self._default_extent()

    def bounding_box(self) -> str:
        """Generate a WSEN bounding box from the geometry.

        Returns:
            Bounding box as comma-delimited WSEN string "w, s, e, n"
        """
        try:
            wkt_string = self._geometry_as_wkt()
            geometry = wkt.loads(wkt_string)
            bounds = geometry.bounds  # Returns (minx, miny, maxx, maxy)
            minx, miny, maxx, maxy = bounds
            return f"{minx}, {miny}, {maxx}, {maxy}"
        except Exception as e:
            logger.warning(f"Error parsing geometry: {self.geom} - {e}")
            return "-180.0, -90.0, 180.0, 90.0"

    def _default_extent(self) -> str:
        """Return default world extent as GeoJSON string.

        Returns:
            JSON string of world extent polygon
        """
        extent = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-180.0, 90.0],
                    [-180.0, -90.0],
                    [180.0, -90.0],
                    [180.0, 90.0],
                    [-180.0, 90.0],
                ]
            ],
        }
        return json.dumps(extent)

    def _envelope_to_polygon(self) -> str:
        """Convert WKT ENVELOPE string to WKT POLYGON string.

        The ENVELOPE format is: ENVELOPE(minx, maxx, maxy, miny)
        Converts to: POLYGON ((minx maxy, minx miny, maxx miny, maxx maxy, minx maxy))

        Returns:
            WKT POLYGON string

        Raises:
            ValueError: If ENVELOPE format is invalid
        """
        # Pattern matches ENVELOPE(minx, maxx, maxy, miny)
        pattern = r"^\s*ENVELOPE\(\s*([-.\d]+)\s*,\s*([-.\d]+)\s*,\s*([-.\d]+)\s*,\s*([-.\d]+)\s*\)\s*$"
        match = re.match(pattern, self.geom, re.IGNORECASE | re.VERBOSE)

        if not match:
            raise ValueError(f"Invalid ENVELOPE format: {self.geom}")

        minx, maxx, maxy, miny = match.groups()
        return f"POLYGON (({minx} {maxy}, {minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}))"

    def _geometry_as_wkt(self) -> str:
        """Return geometry as valid WKT string.

        If the geometry is in ENVELOPE format, convert it to POLYGON.
        Otherwise, return as-is.

        Returns:
            WKT formatted string
        """
        if "ENVELOPE" in self.geom.upper():
            return self._envelope_to_polygon()
        return self.geom
