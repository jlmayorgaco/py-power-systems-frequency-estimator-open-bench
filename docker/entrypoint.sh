#!/usr/bin/env bash
set -euo pipefail

# Re-assert thread pins (allow override via env)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Helpful defaults for logs
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec "$@"
