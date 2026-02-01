-- Add DuckDB native GEOMETRY column from GeoJSON
-- Usage: duckdb -c ".read convert_geometry.sql"

-- Install and load required extensions
INSTALL httpfs;
INSTALL spatial;
LOAD httpfs;
LOAD spatial;

-- Convert ogm.parquet to cloud.parquet with native GEOMETRY
COPY (
  SELECT *,
    ST_GeomFromGeoJSON(geojson) AS geometry
  FROM 'tmp/ogm.parquet'
) TO 'tmp/cloud.parquet' (
  FORMAT PARQUET,
  COMPRESSION zstd,
  PARQUET_VERSION v2
);
