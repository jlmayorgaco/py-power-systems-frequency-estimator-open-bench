# shellcheck shell=bash
# OpenFreqBench — shared environment helpers
# Source from other scripts; do NOT execute directly.
#   source "$ROOT/scripts/common/env.sh"
# Provides:
#   - ENV_NAME / PY_VER defaults (overridable via env)
#   - manager autodetect: micromamba > mamba > conda
#   - with_env <cmd> ...  → run inside the project env
#   - have_env, python_version, timestamp, ensure_dir
#   - cache dirs (OFB_ROOT, NUMBA_CACHE_DIR, MPLCONFIGDIR)
#   - .env loader (dotenv-style, optional)

# --- guard: prevent running directly ---
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file is meant to be sourced, not executed." >&2
  exit 1
fi

# --- minimal logging (kept here so scripts needn't source lib.sh first) ---
ofb_log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
ofb_warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
ofb_err()  { printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }

# --- repo roots (expect caller set ROOT; fallback to two levels up) ---
HERE_guess="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ROOT:=$(cd "$HERE_guess/../.." && pwd)}"

# --- defaults (caller may override via environment) ---
: "${ENV_NAME:=openfreqbench}"
: "${PY_VER:=3.10}"
: "${MANAGER:=}"                 # conda|mamba|micromamba (auto if empty)
: "${OFB_ROOT:=$ROOT/.ofb}"
: "${NUMBA_CACHE_DIR:=$OFB_ROOT/cache/numba}"
: "${MPLCONFIGDIR:=$OFB_ROOT/cache/mpl}"
export OFB_ROOT NUMBA_CACHE_DIR MPLCONFIGDIR

# Create caches silently
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" >/dev/null 2>&1 || true

# --- optional .env loader (dotenv-style) ---
# Looks for .env at repo root; ignores comment lines and blanks.
ofb_load_dotenv() {
  local dotenv="${1:-$ROOT/.env}"
  [[ -f "$dotenv" ]] || return 0
  # shellcheck disable=SC2046
  export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$dotenv" | cut -d= -f1) >/dev/null 2>&1 || true
  # shellcheck source=/dev/null
  set -a; source "$dotenv"; set +a
  ofb_log "Loaded env from $(realpath "$dotenv")"
}

# --- manager detection ---
ofb_pick_manager() {
  if [[ -n "$MANAGER" ]]; then echo "$MANAGER"; return; fi
  if command -v micromamba >/dev/null 2>&1; then echo micromamba; return; fi
  if command -v mamba      >/dev/null 2>&1; then echo mamba; return; fi
  if command -v conda      >/dev/null 2>&1; then echo conda; return; fi
  echo ""
}

MANAGER="$(ofb_pick_manager)"
[[ -z "$MANAGER" ]] && ofb_warn "No conda-like manager found. Install micromamba/mamba/conda."

# --- env presence check (non-fatal) ---
ofb_have_env() {
  case "$MANAGER" in
    micromamba) micromamba env list 2>/dev/null | grep -qE "^[[:space:]]*$ENV_NAME[[:space:]]" ;;
    mamba|conda) conda env list 2>/dev/null | grep -qE "^[[:space:]]*$ENV_NAME[[:space:]]" ;;
    *) return 1 ;;
  esac
}

# --- run inside env (fallback to host if no manager found) ---
with_env() {
  case "$MANAGER" in
    micromamba) micromamba run -n "$ENV_NAME" "$@" ;;
    mamba)      mamba      run -n "$ENV_NAME" "$@" ;;
    conda)      conda      run -n "$ENV_NAME" "$@" ;;
    "")         "$@" ;;  # fallback (warn once)
    *)          ofb_err "Unsupported manager: $MANAGER"; return 127 ;;
  esac
}

# --- tiny helpers ---
ofb_python_version() { with_env python -V 2>&1 || echo "python (unknown)"; }
ofb_timestamp() { date +"%Y%m%d_%H%M%S"; }
ofb_ensure_dir() { mkdir -p -- "$1"; }

# --- optional strictness: require env before continuing (opt-in) ---
# Set OFB_REQUIRE_ENV=1 in caller if you want a hard check.
if [[ "${OFB_REQUIRE_ENV:-0}" -eq 1 ]]; then
  if ! ofb_have_env; then
    ofb_err "Environment '$ENV_NAME' not found. Run scripts/pipelines/install.sh first."
    return 1
  fi
fi
