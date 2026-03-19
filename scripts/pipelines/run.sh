#!/usr/bin/env bash
# OpenFreqBench — pipeline runner
# Uniformly executes Python entrypoints inside the project environment.
# Works with micromamba/mamba/conda (auto-detected).
#
# Examples:
#   scripts/pipelines/run.sh run --scenario s1_synthetic --case frequency_step --est ipdft
#   scripts/pipelines/run.sh list-estimators
#   scripts/pipelines/run.sh list-scenarios
#   scripts/pipelines/run.sh smoke
#
# Env overrides:
#   ENV_NAME=openfreqbench MANAGER=micromamba scripts/pipelines/run.sh run ...

set -Eeuo pipefail

# ---------- Resolve paths ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# ---------- Shared helpers ----------
# logging: log/warn/err/die
if [[ -f "$ROOT/scripts/common/lib.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/lib.sh"
else
  log(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
  warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
  err(){ printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }
  die(){ err "$*"; exit 1; }
fi

# env helpers: with_env, ofb_* utils, caches, manager autodetect
if [[ -f "$ROOT/scripts/common/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/env.sh"
else
  die "Missing $ROOT/scripts/common/env.sh — run installer or add the file."
fi

# Optionally load .env (API keys, etc.)
ofb_load_dotenv || true

# If you want to hard-require the env to exist, uncomment:
# export OFB_REQUIRE_ENV=1
# source "$ROOT/scripts/common/env.sh"

# ---------- Defaults ----------
: "${ENV_NAME:=openfreqbench}"   # already set by env.sh; keep here for clarity

hr(){ printf '%s\n' "-------------------------------------------------------------------------------"; }

usage() {
cat <<'EOF'
OpenFreqBench runner

Commands:
  run                Execute benchmarking pipeline (pipelines.run_benchmark)
  list-estimators    List estimator files/classes (best-effort, no imports)
  list-scenarios     List scenario packages and case functions (best-effort)
  smoke              Tiny end-to-end synthetic sanity check
  help               Show this help

run options:
  --scenario NAME        e.g., s1_synthetic
  --case NAME            e.g., frequency_step | make_clean | ...
  --est NAME             e.g., ipdft | zcd | ...
  --out DIR              Output base dir (default: data/results/bench_<ts>)
  --seed N               RNG seed
  --plots                Save plots (if supported)
  --json                 Save JSON summaries
  --parquet              Save Parquet/HDF
  --verbose              Shell trace for runner
  -- [EXTRA ...]         Extra args passed through to the Python pipeline

Env overrides:
  ENV_NAME, MANAGER (conda|mamba|micromamba), OFB_ROOT
EOF
}

ensure_out_dirs() {
  local base="$1"
  mkdir -p "${base}/"{logs,plots,jsons,metrics} >/dev/null 2>&1 || true
}

# ---------- Introspection (no imports required) ----------
list_estimators() {
  hr; echo "Estimators (files & classes, best-effort)"; hr
  find estimators -type f -name "*.py" ! -name "__init__.py" 2>/dev/null \
    | sed 's#^#  - file: #'
  echo
  if command -v rg >/dev/null 2>&1; then
    rg -n 'class +[A-Za-z0-9_]+\(EstimatorBase\|.*Estimator\)' estimators || true
  else
    grep -RIn 'class .*Estimator' estimators 2>/dev/null || true
  fi
}

list_scenarios() {
  hr; echo "Scenarios (modules & case functions, best-effort)"; hr
  find scenarios -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sed 's#^#  - pkg: #'
  echo
  if command -v rg >/dev/null 2>&1; then
    rg -n 'def +(make_clean|frequency_step|frequency_ramp_step|make_.+|case_.+)\(' scenarios || true
  else
    grep -RIn 'def \(make_clean\|frequency_step\|frequency_ramp_step\|make_.*\|case_.*\)(' scenarios 2>/dev/null || true
  fi
}

# ---------- Smoke test (imports + tiny run) ----------
smoke() {
  hr; echo "Smoke test (synthetic + IpDFT)"; hr
  with_env python - <<'PY'
try:
    from scenarios.s1_synthetic.make_clean import make_clean
    from estimators.basic.ipdft import IpDFT
except Exception as e:
    raise SystemExit(f"Imports failed: {e}")
fs=5000
sig, truth = make_clean(f0=60.0, df=0.2, duration=0.5, fs=fs)
est = IpDFT(fs=fs, frame_len=256)
out = [est.update(x) for x in sig]
print("len:", len(out), "first5:", out[:5], "mean_f:", sum(out)/len(out))
PY
}

# ---------- Run pipeline ----------
run_pipeline() {
  local SCENARIO="" CASE="" EST="" OUTDIR="" SEED=""
  local SAVE_PLOTS=0 SAVE_JSON=0 SAVE_PARQUET=0 VERBOSE=0
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
  [[ -z "$CASE"     ]] && die "Missing --case"
  [[ -z "$EST"      ]] && die "Missing --est"

  [[ -n "$OUTDIR" ]] || OUTDIR="data/results/bench_$(ofb_timestamp)"
  ensure_out_dirs "$OUTDIR"
  [[ "$VERBOSE" -eq 1 ]] && set -x

  # Prefer module runner (pip-installed entrypoint). If you expose a console script
  # 'ofb-run', you can swap to: with_env ofb-run ...
  local PY_ARGS=( -m pipelines.run_benchmark
                  --scenario "$SCENARIO"
                  --case "$CASE"
                  --est "$EST"
                  --out "$OUTDIR" )
  [[ -n "$SEED" ]] && PY_ARGS+=( --seed "$SEED" )
  [[ "$SAVE_PLOTS"   -eq 1 ]] && PY_ARGS+=( --save-plots )
  [[ "$SAVE_JSON"    -eq 1 ]] && PY_ARGS+=( --save-json )
  [[ "$SAVE_PARQUET" -eq 1 ]] && PY_ARGS+=( --save-parquet )
  PY_ARGS+=( "${EXTRA_ARGS[@]}" )

  hr
  echo "Running pipeline"
  hr
  echo "Scenario : $SCENARIO"
  echo "Case     : $CASE"
  echo "Estimator: $EST"
  echo "Output   : $OUTDIR"
  echo "Python   : $(ofb_python_version)"
  echo "Cmd      : python ${PY_ARGS[*]}"
  hr

  local LOG_FILE="${OUTDIR}/logs/run.log"
  ( set -o pipefail; with_env python "${PY_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE" )
  local rc=$?
  [[ $rc -eq 0 ]] || die "Pipeline failed (rc=$rc). See ${LOG_FILE}"
  echo "✅ Done. Logs: ${LOG_FILE}"
}

# ---------- Router ----------
CMD="${1:-help}"; shift || true
case "$CMD" in
  help|-h|--help) usage ;;
  list-estimators) list_estimators ;;
  list-scenarios)  list_scenarios ;;
  smoke)           smoke ;;
  run)             run_pipeline "$@" ;;
  *) err "Unknown command: $CMD"; usage; exit 2;;
esac
