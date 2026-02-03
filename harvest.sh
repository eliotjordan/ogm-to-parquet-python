#!/bin/sh

mkdir tmp
uv run ogm-harvest --download
duckdb -c ".read convert_geometry.sql"
mkdir tmp/data/
mv tmp/cloud.parquet tmp/data/.
mv tmp/ogm-model/embeddings.bin tmp/data/.
mv tmp/ogm-model/tokenizer.json tmp/data/.
