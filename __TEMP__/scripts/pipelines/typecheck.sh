#!/usr/bin/env bash
# OpenFreqBench — static type checking
# Runs mypy (and optionally pyright if present) inside the project environment.
#
# Examples:
#   scripts/pipelines/typecheck.sh
#   scripts/pipelines/typecheck.sh --since origin/main
#   scripts/pipelines/typecheck.sh --staged
#   scripts/pipelines/typecheck.sh --all --strict
#   scripts/pipelines/typecheck.sh --paths estimators pipelines
#
# Exit codes: nonzero if any checker fails.

set -Eeuo pipefail

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
  die "Missing $ROOT/scripts/common/env.sh — run installer or add the file."
fi

# (Optional) load .env, though not typically needed for typecheck
ofb_load_dotenv || true

# ---------- Defaults & args ----------
STRICT=0
PYRIGHT=0           # also run pyright if available or forced
SINCE=""
STAGED=0
ALL=0
CLEAR_CACHE=0
SHOW_CONFIG=0
BASELINE=0          # mypy --any-exprs-report to seed a baseline dir
REPORT_DIR_DEFAULT="$OFB_ROOT/typecheck"
declare -a USER_PATHS=()
declare -a EXTRA_MYPY_ARGS=()

usage() {
  cat <<'EOF'
OpenFreqBench typecheck

Selection:
  --all                 Check entire repo (default paths)
  --staged              Only files touching staged changes (git)
  --since <ref>         Only files changed since <ref> (e.g., origin/main)
  --paths <...>         Explicit paths (rest of args)

Behavior:
  --strict              Add stricter mypy flags (on top of pyproject config)
  --clear-cache         Clear .mypy_cache before running
  --show-config         Print mypy config resolution and exit
  --baseline            Emit mypy "any-exprs" report into .ofb/typecheck/any/
  --pyright             Also run pyright if installed (or fail if not)

Other:
  -- <ARGS>             Pass additional args to mypy verbatim
  -h, --help            Show help
EOF
}

while (( "$#" )); do
  case "$1" in
    --all) ALL=1; shift ;;
    --staged) STAGED=1; shift ;;
    --since) SINCE="$2"; shift 2 ;;
    --paths) shift; USER_PATHS=("$@"); break ;;
    --strict) STRICT=1; shift ;;
    --clear-cache) CLEAR_CACHE=1; shift ;;
    --show-config) SHOW_CONFIG=1; shift ;;
    --baseline) BASELINE=1; shift ;;
    --pyright) PYRIGHT=1; shift ;;
    --) shift; EXTRA_MYPY_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) warn "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- Targets ----------
git_available() { command -v git >/dev/null 2>&1 && [[ -d .git ]]; }

default_paths=( "openfreqbench" "estimators" "scenarios" "evaluation" "pipelines" "tests" )
paths=()

if (( ${#USER_PATHS[@]} )); then
  paths=("${USER_PATHS[@]}")
elif [[ $ALL -eq 1 ]]; then
  paths=("${default_paths[@]}")
elif [[ -n "$SINCE" ]] && git_available; then
  mapfile -t changed < <(git diff --name-only --diff-filter=ACMRT "$SINCE"... -- '*.py' || true)
  paths=("${changed[@]}")
elif [[ $STAGED -eq 1 ]] && git_available; then
  mapfile -t staged < <(git diff --name-only --cached --diff-filter=ACMRT -- '*.py' || true)
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

# Keep only existing files/dirs
clean_paths=()
for p in "${paths[@]}"; do
  [[ -e "$p" ]] && clean_paths+=("$p")
done
paths=("${clean_paths[@]}")

if [[ ${#paths[@]} -eq 0 ]]; then
  log "No Python files to type-check. Exiting."
  exit 0
fi

log "Type-check targets: ${paths[*]}"

# ---------- Logs / reports ----------
LOG_DIR="$OFB_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/typecheck-$(ofb_timestamp).log"
log "Logs: $LOG_FILE"

[[ $CLEAR_CACHE -eq 1 ]] && { log "Clearing .mypy_cache"; rm -rf .mypy_cache || true; }

# ---------- Build mypy args ----------
mypy_args=()
(( STRICT )) && mypy_args+=("--warn-unused-ignores" "--no-implicit-optional" "--warn-redundant-casts")
(( SHOW_CONFIG )) && mypy_args+=("--show-config")  # will exit after printing

# Baseline report (useful to track Any usage reduction)
if [[ $BASELINE -eq 1 ]]; then
  REPORT_DIR="${REPORT_DIR_DEFAULT}/any"
  mkdir -p "$REPORT_DIR"
  mypy_args+=("--any-exprs-report" "$REPORT_DIR")
  log "Baseline report will be in: $REPORT_DIR"
fi

# Extra args passthrough
mypy_args+=("${EXTRA_MYPY_ARGS[@]}")
# Targets at the end
mypy_args+=("${paths[@]}")

# ---------- Run mypy ----------
status=0
( set -o pipefail; with_env mypy "${mypy_args[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?

# ---------- Optional pyright ----------
if [[ $PYRIGHT -eq 1 ]]; then
  if with_env python -c "import shutil,sys;sys.exit(0 if shutil.which('pyright') else 1)" >/dev/null 2>&1; then
    log "Running pyright"
    ( set -o pipefail; with_env pyright "${paths[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?
  else
    err "pyright not found in environment."
    status=2
  fi
fi

if [[ $status -ne 0 ]]; then
  err "❌ Type-check failed (code=$status). See $LOG_FILE"
  exit "$status"
fi

log "✅ Type-check passed."
