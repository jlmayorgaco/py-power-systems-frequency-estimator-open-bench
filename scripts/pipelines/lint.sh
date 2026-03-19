#!/usr/bin/env bash
# OpenFreqBench — lint & type-check pipeline
# Runs Ruff (format/lint) and Mypy inside the project environment.

set -Eeuo pipefail

# --- Bash 3.2 compatibility (macOS): mapfile/readarray shim ---
if ! command -v mapfile >/dev/null 2>&1; then
  mapfile() {
    local opt strip_newline=0
    while getopts ":t" opt; do
      case "$opt" in t) strip_newline=1;; esac
    done
    shift $((OPTIND - 1))
    local arr_name="${1:-}"
    [[ -z "$arr_name" ]] && { echo "mapfile shim: missing array name" >&2; return 1; }
    eval "$arr_name=()"
    local line
    while IFS= read -r line; do
      eval "$arr_name+=(\"\$line\")"
    done
  }
  readarray() { mapfile "$@"; }
fi

# ---------- Resolve paths ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# ---------- Shared helpers ----------
if [[ -f "$ROOT/scripts/common/lib.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/lib.sh"
else
  log(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
  warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
  err(){ printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }
  die(){ err "$*"; exit 1; }
fi

if [[ -f "$ROOT/scripts/common/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/env.sh"
else
  # Fallback: assume we're already inside the env; just exec directly
  with_env() { "$@"; }
  ofb_load_dotenv(){ :; }
fi

# Optionally load .env (not required for linting)
ofb_load_dotenv || true

# ---------- Defaults & args ----------
FIX=0
FORMAT_ONLY=0
STRICT=0
RUN_MYPY=1
RUN_RUFF=1
STAGED=0
ALL=0
SINCE=""
declare -a USER_PATHS=()

usage() {
  cat <<'EOF'
OpenFreqBench lint

Flags:
  --fix             Run Ruff with autofix (ruff check --fix), then mypy
  --format-only     Only run 'ruff format' (no lint/mypy)
  --no-mypy         Skip mypy
  --no-ruff         Skip Ruff (lint/format)
  --strict          Stricter settings (tool-config dependent)
  --staged          Only lint staged files (git)
  --since <ref>     Only lint files changed since <ref> (e.g., main)
  --all             Lint entire repo (default paths)
  --paths <...>     Paths/files to lint (space-separated; end of args)
  -h, --help        Show help
EOF
}

while (( "$#" )); do
  case "$1" in
    --fix) FIX=1; shift ;;
    --format-only) FORMAT_ONLY=1; shift ;;
    --no-mypy) RUN_MYPY=0; shift ;;
    --no-ruff) RUN_RUFF=0; shift ;;
    --strict) STRICT=1; shift ;;
    --staged) STAGED=1; shift ;;
    --since) SINCE="$2"; shift 2 ;;
    --all) ALL=1; shift ;;
    --paths) shift; USER_PATHS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) warn "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- Target discovery ----------
git_available() { command -v git >/dev/null 2>&1 && [[ -d .git ]]; }

default_paths=( "openfreqbench" "estimators" "scenarios" "evaluation" "pipelines" "tests" )
paths=()

if (( ${#USER_PATHS[@]} )); then
  paths=("${USER_PATHS[@]}")
elif [[ $ALL -eq 1 ]]; then
  paths=("${default_paths[@]}")
elif [[ -n "${SINCE:-}" ]] && git_available; then
  mapfile -t changed < <(git diff --name-only --diff-filter=ACMRT "$SINCE"... -- '*.py' || true)
  paths=("${changed[@]}")
elif [[ $STAGED -eq 1 ]] && git_available; then
  mapfile -t staged  < <(git diff --name-only --cached --diff-filter=ACMRT -- '*.py' || true)
  paths=("${staged[@]}")
else
  if git_available; then
    mapfile -t changed < <(git diff --name-only --diff-filter=ACMRT HEAD -- '*.py' || true)
    if [[ ${#changed[@]} -gt 0 ]]; then
      paths=("${changed[@]}")
    else
      paths=("${default_paths[@]}")
    fi
  else
    paths=("${default_paths[@]}")
  fi
fi

# Filter out non-existing paths
clean_paths=()
for p in "${paths[@]}"; do
  [[ -e "$p" ]] && clean_paths+=("$p")
done
paths=("${clean_paths[@]}")

if [[ ${#paths[@]} -eq 0 ]]; then
  log "No Python files to lint. Exiting."
  exit 0
fi

log "Lint targets: ${paths[*]}"

LOG_DIR="${OFB_ROOT:-$ROOT/.ofb}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lint-$(date -u +%Y%m%dT%H%M%SZ).log"
log "Logs: $LOG_FILE"

# ---------- Ensure tools present (auto-install if missing) ----------
ensure_ruff() {
  [[ $RUN_RUFF -eq 1 ]] || return 0
  if ! with_env ruff --version >/dev/null 2>&1; then
    log "Installing Ruff (dev tool missing)…"
    with_env python -m pip install --quiet ruff || die "Failed to install Ruff"
  fi
}
ensure_mypy() {
  [[ $RUN_MYPY -eq 1 ]] || return 0
  if ! with_env mypy --version >/dev/null 2>&1; then
    log "Installing Mypy (dev tool missing)…"
    with_env python -m pip install --quiet mypy || die "Failed to install Mypy"
  fi
}
ensure_ruff
ensure_mypy

# ---------- Steps ----------
status=0

run_ruff_format() {
  [[ $RUN_RUFF -eq 1 ]] || return 0
  log "Ruff format"
  ( set -o pipefail; with_env ruff format "${paths[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?
}

run_ruff_check() {
  [[ $RUN_RUFF -eq 1 ]] || return 0
  local args=(check)
  [[ $FIX -eq 1 ]] && args+=(--fix)
  [[ $STRICT -eq 1 ]] && args+=(--unsafe-fixes)
  log "Ruff ${args[*]-}"
  ( set -o pipefail; with_env ruff "${args[@]}" "${paths[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?
}

# --- replace your existing run_mypy() with this ---
run_mypy() {
  [[ $RUN_MYPY -eq 1 ]] || return 0
  local args=()
  [[ $STRICT -eq 1 ]] && args+=(--warn-unused-ignores --no-warn-no-return)

  # Safe log under set -u even if args is empty
  log "Mypy ${args[*]-}"

  # Avoid empty-array expansion on Bash 3.2 + set -u
  if [[ ${#args[@]} -gt 0 ]]; then
    ( set -o pipefail; with_env mypy "${args[@]}" "${paths[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?
  else
    ( set -o pipefail; with_env mypy "${paths[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?
  fi
}


# ---------- Execute ----------
if [[ $FORMAT_ONLY -eq 1 ]]; then
  run_ruff_format
else
  run_ruff_format
  run_ruff_check
  run_mypy
fi

if [[ $status -ne 0 ]]; then
  err "❌ Lint pipeline failed (code=$status). See $LOG_FILE"
  exit "$status"
fi

log "✅ Lint pipeline passed."
