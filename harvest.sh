#!/bin/sh

mkdir tmp
uv run ogm-harvest --download --no-embeddings
duckdb -c ".read convert_geometry.sql"
