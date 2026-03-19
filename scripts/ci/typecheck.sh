#!/usr/bin/env bash
set -euo pipefail
python -m mypy src/openfreqbench --ignore-missing-imports
