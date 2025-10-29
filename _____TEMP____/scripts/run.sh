#!/usr/bin/env bash
# OpenFreqBench runner (macOS/Linux)
# Usage examples:
#   scripts/run.sh run --scenario s1_synthetic --case frequency_step --est ipdft
#   scripts/run.sh list-estimators
#   scripts/run.sh list-scenarios
#   scripts/run.sh smoke
set -euo pipefail

# Centralize caches
export OFB_ROOT="${OFB_ROOT:-.ofb}"
export NUMBA_CACHE_DIR="${OFB_ROOT}/cache/numba"
export MPLCONFIGDIR="${OFB_ROOT}/cache/mpl"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# -------- repo + env detection -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="openfreqbench"

choose_runner() {
  if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "conda run -n ${ENV_NAME} python"
  elif command -v micromamba >/dev/null 2>&1 && micromamba env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "micromamba run -n ${ENV_NAME} python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}
PYBIN="$(choose_runner)"

# -------- helpers --------------------------------------------------------------
hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }
die() { echo "❌ $*" >&2; exit 1; }
info() { echo "➜ $*"; }

usage() {
  cat <<'EOF'
OpenFreqBench runner

Commands:
  run                Execute benchmarking pipeline
  list-estimators    Show available estimator modules/classes (best-effort)
  list-scenarios     Show available scenarios & cases (best-effort)
  smoke              Tiny end-to-end synthetic check
  help               This help

run options (common):
  --scenario NAME        Scenario package under scenarios/ (e.g., s1_synthetic)
  --case NAME            Scenario case/function (e.g., frequency_step, make_clean)
  --est NAME             Estimator id/module (e.g., ipdft, zcd)
  --out DIR              Output base directory (default: data/results/<timestamp>)
  --seed N               RNG seed (optional)
  --plots                Save plots (if pipeline supports it)
  --json                 Save JSON summaries (if pipeline supports it)
  --parquet              Save Parquet/HDF (if pipeline supports it)
  --verbose              Shell debug (set -x)
  -- [EXTRA ...]         Extra args passed through to the Python pipeline

Examples:
  scripts/run.sh run --scenario s1_synthetic --case frequency_step --est ipdft --plots --json
  scripts/run.sh run --scenario s1_synthetic --case make_clean --est zcd --out data/results/dev -- --frame-len 256
EOF
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

ensure_out_dirs() {
  local base="$1"
  mkdir -p "${base}/"{logs,plots,jsons,metrics}
}

# -------- introspection (best-effort, no imports required) ---------------------
list_estimators() {
  hr; echo "Estimators (files & classes, best-effort)"; hr
  # list files
  find estimators -type f -name "*.py" ! -name "__init__.py" \
    | sed 's#^#  - file: #'
  echo
  # grep classes
  if command -v rg >/dev/null 2>&1; then
    rg -n 'class +[A-Za-z0-9_]+\(EstimatorBase\|.*Estimator\)' estimators || true
  else
    grep -RIn 'class .*Estimator' estimators 2>/dev/null || true
  fi
}

list_scenarios() {
  hr; echo "Scenarios (modules & case functions, best-effort)"; hr
  find scenarios -type d -maxdepth 1 -mindepth 1 | sed 's#^#  - pkg: #'
  echo
  if command -v rg >/dev/null 2>&1; then
    rg -n 'def +(make_clean|frequency_step|frequency_ramp_step|make_.+|case_.+)\(' scenarios || true
  else
    grep -RIn 'def \(make_clean\|frequency_step\|frequency_ramp_step\|make_.*\|case_.*\)(' scenarios 2>/dev/null || true
  fi
}

# -------- smoke test -----------------------------------------------------------
smoke() {
  hr; echo "Smoke test (synthetic + IpDFT)"; hr
  "$PYBIN" - <<'PY'
from scenarios.s1_synthetic.make_clean import make_clean
from estimators.basic.ipdft import IpDFT

fs=5000
sig, truth = make_clean(f0=60.0, df=0.2, duration=0.5, fs=fs)
est = IpDFT(fs=fs, frame_len=256)
out = [est.update(x) for x in sig]
print("len:", len(out), "first5:", out[:5], "mean_f:", sum(out)/len(out))
PY
}

# -------- run command ----------------------------------------------------------
run_pipeline() {
  local SCENARIO="" CASE="" EST="" OUTDIR="" SEED="" VERBOSE=0
  local SAVE_PLOTS=0 SAVE_JSON=0 SAVE_PARQUET=0
  local EXTRA_ARGS=()

  while (( "$#" )); do
    case "$1" in
      --scenario) SCENARIO="$2"; shift 2;;
      --case) CASE="$2"; shift 2;;
      --est) EST="$2"; shift 2;;
      --out) OUTDIR="$2"; shift 2;;
      --seed) SEED="$2"; shift 2;;
      --plots) SAVE_PLOTS=1; shift;;
      --json) SAVE_JSON=1; shift;;
      --parquet) SAVE_PARQUET=1; shift;;
      --verbose) VERBOSE=1; shift;;
      --) shift; EXTRA_ARGS+=("$@"); break;;
      -h|--help) usage; exit 0;;
      *) EXTRA_ARGS+=("$1"); shift;;
    esac
  done

  [[ -z "$SCENARIO" ]] && die "Missing --scenario"
  [[ -z "$CASE" ]] && die "Missing --case"
  [[ -z "$EST" ]] && die "Missing --est"

  if [[ -z "${OUTDIR}" ]]; then
    OUTDIR="data/results/bench_$(timestamp)"
  fi
  ensure_out_dirs "$OUTDIR"

  [[ "$VERBOSE" -eq 1 ]] && set -x

  # Build flags for your Python CLI (best-guess; adjust to your argparse names)
  PY_ARGS=( -m pipelines.run_benchmark
            --scenario "$SCENARIO"
            --case "$CASE"
            --est "$EST"
            --out "$OUTDIR" )
  [[ -n "$SEED" ]] && PY_ARGS+=( --seed "$SEED" )
  [[ "$SAVE_PLOTS" -eq 1 ]] && PY_ARGS+=( --save-plots )
  [[ "$SAVE_JSON" -eq 1 ]] && PY_ARGS+=( --save-json )
  [[ "$SAVE_PARQUET" -eq 1 ]] && PY_ARGS+=( --save-parquet )
  # Pass-through extras
  PY_ARGS+=( "${EXTRA_ARGS[@]}" )

  hr; echo "Running pipeline"; hr
  echo "Scenario : $SCENARIO"
  echo "Case     : $CASE"
  echo "Estimator: $EST"
  echo "Output   : $OUTDIR"
  echo "Python   : $($PYBIN -V 2>&1)"
  echo "Cmd      : $PYBIN ${PY_ARGS[*]}"
  hr

  # Log to file and console
  LOG_FILE="${OUTDIR}/logs/run.log"
  ( set -o pipefail; "$PYBIN" "${PY_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE" )
  echo "✅ Done. Logs: ${LOG_FILE}"
}

# -------- command router -------------------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
  help|-h|--help) usage ;;
  list-estimators) list_estimators ;;
  list-scenarios)  list_scenarios ;;
  smoke)           smoke ;;
  run)             run_pipeline "$@" ;;
  *)
    echo "Unknown command: $CMD"
    usage
    exit 2
    ;;
esac
