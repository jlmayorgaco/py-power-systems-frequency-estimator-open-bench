#!/usr/bin/env bash
set -euo pipefail
source .venv-ofb/bin/activate
pytest tests/smoke/ -v --tb=short
