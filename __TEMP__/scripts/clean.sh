#!/usr/bin/env bash
set -euo pipefail
rm -rf .ofb \
       __pycache__ */__pycache__ \
       .pytest_cache .ruff_cache .mypy_cache \
       *.egg-info build dist
echo "🧹 Cleaned caches and build artifacts."
