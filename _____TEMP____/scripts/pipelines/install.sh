#!/usr/bin/env bash
# OpenFreqBench — environment installer (ALL deps by default)
# Creates/updates a conda-like env and installs project in editable mode.
# Now installs dev tools by default (ruff, mypy, pre-commit, pytest, pytest-cov).
#
# Usage:
#   scripts/pipelines/install.sh [--extras opendss,viz,notebooks] \
#                                [--python 3.10] [--env openfreqbench] \
#                                [--manager conda|mamba|micromamba]
#
# Examples:
#   scripts/pipelines/install.sh --extras opendss,viz
#   scripts/pipelines/install.sh --python 3.12 --env ofb-312
#
set -Eeuo pipefail

# ---------- Resolve paths ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# ---------- Colors / styling ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
BLUE=$'\033[34m'; CYAN=$'\033[36m'; MAGENTA=$'\033[35m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'

# ---------- Shared helpers (fallbacks if lib.sh not present) ----------
if [[ -f "$ROOT/scripts/common/lib.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/lib.sh"
else
  log(){ printf "%s %s\n" "${BLUE}[INFO]${RESET}" "$*"; }
  warn(){ printf "%s %s\n" "${YELLOW}[WARN]${RESET}" "$*"; }
  err(){ printf "%s  %s\n" "${RED}[ERR ]${RESET}" "$*" >&2; }
  die(){ err "$*"; exit 1; }
  req(){ command -v "$1" >/dev/null || die "Missing required tool: $1"; }
fi

# ---------- Pretty banner helpers (ASCII for portability) ----------
_termw() { local COLUMNS=${COLUMNS:-$(tput cols 2>/dev/null || echo 80)}; echo "$COLUMNS"; }
_hr()    { printf '%*s\n' "$(_termw)" '' | tr ' ' '-'; }
_center() {
  local text="$1" w; w="$(_termw)"; local pad=$(( ( ${#text} + w ) / 2 )); printf "%*s\n" "$pad" "$text"
}
banner_start() {
  echo; echo
  _hr; _hr
  _center "${BOLD}Py Power Systems Frequency Estimator — Open Bench${RESET}"
  _center "${DIM}scripts/pipelines/install.sh${RESET}"
  _hr; echo
}
banner_context() {
  local env="$1" py="$2" mgr="$3" extras="$4"
  printf "%s\n" "${DIM}Context:${RESET}"
  printf "  • Env Name   : %s%s%s\n" "${BOLD}" "$env" "${RESET}"
  printf "  • Python     : %s%s%s\n" "${BOLD}" "$py"  "${RESET}"
  printf "  • Manager    : %s%s%s\n" "${BOLD}" "$mgr" "${RESET}"
  printf "  • Extras     : %s%s%s\n" "${BOLD}" "${extras:-<none>}" "${RESET}"
  echo
}
banner_success() {
  echo
  _hr
  _center "${GREEN}${BOLD}✅ Environment ready${RESET}"
  _hr
  echo
}

# ---------- Defaults ----------
ENV_NAME="${ENV_NAME:-openfreqbench}"
PY_VER="${PY_VER:-3.10}"
EXTRAS="${EXTRAS-}"       # optional comma list: opendss,viz,notebooks
MANAGER="${MANAGER-}"     # conda|mamba|micromamba (auto if empty)

# ---------- Args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras) EXTRAS="${2:-}"; shift 2 ;;
    --python) PY_VER="${2:-}"; shift 2 ;;
    --env) ENV_NAME="${2:-}"; shift 2 ;;
    --manager) MANAGER="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: ${0##*/} [--extras opendss,viz,notebooks] [--python 3.10] [--env NAME] [--manager conda|mamba|micromamba]
EOF
      exit 0
      ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# ---------- Banner (header) ----------
banner_start

# ---------- Choose environment manager ----------
pick_manager() {
  if [[ -n "$MANAGER" ]]; then echo "$MANAGER"; return; fi
  if command -v micromamba >/dev/null 2>&1; then echo "micromamba"; return; fi
  if command -v mamba      >/dev/null 2>&1; then echo "mamba"; return; fi
  if command -v conda      >/dev/null 2>&1; then echo "conda"; return; fi
  die "No conda-compatible manager found. Install micromamba, mamba, or conda."
}
MANAGER="$(pick_manager)"
log "Env manager: $MANAGER"

# ---------- Make 'conda' usable in non-interactive shells ----------
if [[ "$MANAGER" == "conda" ]]; then
  # shellcheck disable=SC1091
  for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" \
           "/opt/homebrew/Caskroom/mambaforge/base/etc/profile.d/conda.sh" \
           "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
    [[ -r "$f" ]] && source "$f" && break
  done
fi

# ---------- Show context box ----------
banner_context "$ENV_NAME" "$PY_VER" "$MANAGER" "$EXTRAS"

# Wrapper to exec commands inside the env (uniform across managers)
with_env() {
  case "$MANAGER" in
    micromamba) micromamba run -n "$ENV_NAME" "$@";;
    mamba)      mamba      run -n "$ENV_NAME" "$@";;
    conda)      conda      run -n "$ENV_NAME" "$@";;
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

# ---------- Build extras list (Bash 3.2 + set -u safe, no empty tokens) ----------
declare -a EXTRA_CANDIDATES=()   # we always install dev tools below; extras remain optional
declare -a EXTRAS_ARR=()
if [[ -n "${EXTRAS-}" ]]; then
  while IFS= read -r e; do
    e="$(printf "%s" "$e" | awk '{$1=$1}1')"   # trim whitespace
    EXTRAS_ARR+=("$e")
  done <<EOF
$(printf "%s" "${EXTRAS//,/\\n}")
EOF
fi

EXTRA_LIST=()
for tok in "${EXTRA_CANDIDATES[@]:-}" "${EXTRAS_ARR[@]:-}"; do
  tok="$(printf "%s" "$tok" | awk '{$1=$1}1')"
  [[ -n "$tok" ]] && EXTRA_LIST+=("$tok")
done

if [[ ${#EXTRA_LIST[@]} -gt 0 ]]; then
  EXTRAS_STR="$(IFS=,; echo "${EXTRA_LIST[*]}")"
  log "Installing project in editable mode with extras: [${EXTRAS_STR}]"
  with_env python -m pip install -e ".[${EXTRAS_STR}]"
else
  log "Installing project in editable mode (no extras)"
  with_env python -m pip install -e .
fi

# ---------- Always install dev tools (no flag needed) ----------
log "Installing developer tools (ruff, mypy, pre-commit, pytest, pytest-cov)"
with_env python -m pip install -U ruff mypy pre-commit pytest pytest-cov

# ---------- Optional runtime libs via pip (skip if already present) ----------
log "Installing optional runtime libs via pip (if missing)"
with_env python - <<'PY'
import sys, subprocess, importlib
OPTIONALS = {
    "opendssdirect.py": "opendssdirect",
    "plotly": "plotly",
    "joblib": "joblib",
    "brokenaxes": "brokenaxes",
    "ssqueezepy": "ssqueezepy",
}
for pip_name, import_name in OPTIONALS.items():
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
PY

# ---------- Dev niceties ----------
if command -v git >/dev/null 2>&1 && [[ -d "$ROOT/.git" ]]; then
  log "Installing pre-commit hooks"
  with_env pre-commit install || true
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
        print("  ✅", m)
    except Exception as e:
        ok = False
        print("  ❌", m, "->", e)
raise SystemExit(0 if ok else 1)
PY
rc=$?
set -e
[[ $rc -eq 0 ]] || die "Smoke test failed."

# ---------- Success / next steps ----------
banner_success
printf "Activate: %s\n" "$([[ $MANAGER == micromamba ]] && echo "micromamba activate ${ENV_NAME}" || echo "conda activate ${ENV_NAME}")"
echo
echo "Next steps:"
printf "  • %sRun smoke%s  : make smoke\n"  "$BOLD" "$RESET"
printf "  • %sRun lint%s   : make lint\n"   "$BOLD" "$RESET"
printf "  • %sRun tests%s  : make test-all\n" "$BOLD" "$RESET"
printf "  • %sDocs%s       : make docs-serve\n" "$BOLD" "$RESET"
echo
printf "%sTip:%s run inside env without activating:\n" "$BOLD" "$RESET"
printf "  %s run -n %s python -m pip list\n" "$MANAGER" "$ENV_NAME"
exit 0
