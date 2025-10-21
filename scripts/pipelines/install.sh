#!/usr/bin/env bash
# OpenFreqBench — environment installer
# Creates/updates a conda-like env and installs project in editable mode.
# Usage:
#   scripts/pipelines/install.sh [--dev] [--extras opendss,viz,notebooks] [--python 3.10] [--env openfreqbench] [--manager conda|mamba|micromamba]
#
# Examples:
#   scripts/pipelines/install.sh --dev --extras opendss,viz
#   scripts/pipelines/install.sh --python 3.12 --env ofb-312

set -Eeuo pipefail

# ---------- Resolve paths ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Optional shared helpers (logging, die, req). If missing, define fallbacks.
if [[ -f "$ROOT/scripts/common/lib.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/lib.sh"
else
  log(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
  warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
  err(){ printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }
  die(){ err "$*"; exit 1; }
  req(){ command -v "$1" >/dev/null || die "Missing required tool: $1"; }
fi

# ---------- Defaults ----------
ENV_NAME="${ENV_NAME:-openfreqbench}"
PY_VER="${PY_VER:-3.10}"
WITH_DEV=0
EXTRAS=""                 # comma-separated extras: opendss,viz,notebooks
MANAGER=""                # conda|mamba|micromamba (auto if empty)

# ---------- Args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) WITH_DEV=1; shift ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --python) PY_VER="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --manager) MANAGER="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: ${0##*/} [--dev] [--extras opendss,viz,notebooks] [--python 3.10] [--env NAME] [--manager conda|mamba|micromamba]
EOF
      exit 0
      ;;
    *) die "Unknown arg: $1" ;;
  endcase
done

# ---------- Choose environment manager ----------
pick_manager() {
  if [[ -n "$MANAGER" ]]; then echo "$MANAGER"; return; fi
  if command -v micromamba >/dev/null 2>&1; then echo "micromamba"; return; fi
  if command -v mamba >/dev/null 2>&1; then echo "mamba"; return; fi
  if command -v conda >/dev/null 2>&1; then echo "conda"; return; fi
  die "No conda-compatible manager found. Install micromamba, mamba, or conda."
}
MANAGER="$(pick_manager)"
log "Env manager: $MANAGER"

# Wrapper to exec commands inside the env (uniform across managers)
with_env() {
  case "$MANAGER" in
    micromamba) micromamba run -n "$ENV_NAME" "$@";;
    mamba)      mamba run      -n "$ENV_NAME" "$@";;
    conda)      conda run      -n "$ENV_NAME" "$@";;
    *) die "Unsupported manager: $MANAGER";;
  esac
}

# Manager-specific create/install subcommands
mm_create_env() {
  case "$MANAGER" in
    micromamba) micromamba create -y -n "$ENV_NAME" -c conda-forge "python=${PY_VER}" ;;
    mamba)      mamba      create -y -n "$ENV_NAME" -c conda-forge "python=${PY_VER}" ;;
    conda)      conda      create -y -n "$ENV_NAME" -c conda-forge "python=${PY_VER}" ;;
  esac
}

mm_install_pkgs() {
  case "$MANAGER" in
    micromamba) micromamba install -y -n "$ENV_NAME" -c conda-forge "$@" ;;
    mamba)      mamba      install -y -n "$ENV_NAME" -c conda-forge "$@" ;;
    conda)      conda      install -y -n "$ENV_NAME" -c conda-forge "$@" ;;
  esac
}

# ---------- Caches ----------
export OFB_ROOT="${OFB_ROOT:-$ROOT/.ofb}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$OFB_ROOT/cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OFB_ROOT/cache/mpl}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# ---------- Create/Update env ----------
log "Ensuring env: ${ENV_NAME} (python=${PY_VER})"
mm_create_env || true

log "Installing base scientific stack (via conda-forge)"
mm_install_pkgs numpy scipy pandas matplotlib numba pywavelets jupyter tqdm pyyaml pip

# ---------- Pip install project ----------
export PIP_DISABLE_PIP_VERSION_CHECK=1
log "Upgrading pip & wheel"
with_env python -m pip install -U pip wheel

# Build extras list: dev (+ any user extras)
EXTRA_LIST=()
if [[ $WITH_DEV -eq 1 ]]; then EXTRA_LIST+=("dev"); fi
# add any comma-separated extras
IFS=',' read -r -a USER_EXTRAS <<< "${EXTRAS:-}"
for e in "${USER_EXTRAS[@]}"; do
  [[ -n "$e" ]] && EXTRA_LIST+=("$e")
done

if [[ ${#EXTRA_LIST[@]} -gt 0 ]]; then
  EXTRAS_STR="$(IFS=,; echo "${EXTRA_LIST[*]}")"
  log "Installing project in editable mode with extras: [${EXTRAS_STR}]"
  with_env python -m pip install -e ".[${EXTRAS_STR}]"
else
  log "Installing project in editable mode (no extras)"
  with_env python -m pip install -e .
fi

# Optional: packages that aren’t reliable on conda
# (Keep lean; many are already covered by extras)
log "Installing optional runtime libs via pip (if missing)"
with_env python - <<'PY'
import sys, subprocess
pkgs = ["opendssdirect.py","plotly","joblib","brokenaxes","ssqueezepy"]
for p in pkgs:
    try:
        __import__(p.split("==")[0].split(">=")[0].replace("-", "_"))
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])
PY

# ---------- Dev niceties ----------
if [[ $WITH_DEV -eq 1 ]]; then
  if command -v git >/dev/null 2>&1 && [[ -d "$ROOT/.git" ]]; then
    log "Installing pre-commit hooks"
    with_env python -m pip install pre-commit
    with_env pre-commit install || true
  fi
fi

# ---------- Smoke test ----------
log "Running smoke test (imports)"
set +e
with_env python - <<'PY'
mods = ["numpy","scipy","pandas","matplotlib","numba","pywt","tqdm","opendssdirect"]
ok = True
for m in mods:
    try:
        __import__(m)
        print(f"  ✅ {m}")
    except Exception as e:
        ok = False
        print(f"  ❌ {m}: {e}")
raise SystemExit(0 if ok else 1)
PY
rc=$?
set -e
[[ $rc -eq 0 ]] || die "Smoke test failed."

log "✅ Environment ready."
echo
echo "Activate shell env (interactive):"
case "$MANAGER" in
  micromamba) echo "  micromamba activate ${ENV_NAME}" ;;
  mamba|conda) echo "  conda activate ${ENV_NAME}" ;;
esac
echo
echo "Run any command inside the env without activating:"
echo "  $(basename "$MANAGER") run -n ${ENV_NAME} python -m pip list"
echo
