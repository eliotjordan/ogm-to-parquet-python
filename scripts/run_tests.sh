#!/usr/bin/env bash
# Run tests in separate groups to avoid segfault from C library conflicts
# (pyvips + model2vec on macOS). Coverage is accumulated across runs.

set -e

# Clean previous coverage data
rm -f .coverage .coverage.*

echo "=== Running test group 1: derivatives, download, enrichment, geometry, text_extract ==="
uv run pytest tests/test_derivatives.py tests/test_download.py tests/test_enrichment.py tests/test_geometry.py tests/test_text_extract.py \
    --cov --cov-report= "$@"

echo ""
echo "=== Running test group 2: embeddings, harvest ==="
uv run pytest tests/test_embeddings.py tests/test_harvest.py \
    --cov --cov-append --cov-report= "$@"

echo ""
echo "=== Combined Coverage Report ==="
uv run coverage report --show-missing

# Generate HTML report if requested
if [[ "$*" == *"--cov-report=html"* ]] || [[ -z "$*" ]]; then
    uv run coverage html
    echo "HTML coverage report written to htmlcov/"
fi
