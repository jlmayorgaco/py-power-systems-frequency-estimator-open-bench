#!/usr/bin/env bash
set -euo pipefail
echo "==> Cleaning build artifacts and caches"
rm -rf .venv-ofb *.egg-info build dist .pytest_cache .mypy_cache .ruff_cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "==> Clean complete"
