#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/ -v --tb=short --cov=src/openfreqbench --cov-report=xml
