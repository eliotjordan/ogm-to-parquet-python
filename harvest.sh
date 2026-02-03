#!/bin/sh

mkdir tmp
uv run ogm-harvest --download
duckdb -c ".read convert_geometry.sql"
