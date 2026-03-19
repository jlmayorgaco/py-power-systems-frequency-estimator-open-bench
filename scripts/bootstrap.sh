#!/usr/bin/env bash
set -euo pipefail
echo "==> Setting up OpenFreqBench development environment"
python3 -m venv .venv-ofb
source .venv-ofb/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
echo "==> Done. Activate with: source .venv-ofb/bin/activate"
