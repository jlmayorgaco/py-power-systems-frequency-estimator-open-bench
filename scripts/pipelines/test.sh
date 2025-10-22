#!/usr/bin/env bash
# OpenFreqBench — test runner (pytest + coverage)
# Examples:
#   scripts/pipelines/test.sh
#   scripts/pipelines/test.sh --since main -k ipdft
#   scripts/pipelines/test.sh --markers "not slow" --workers auto --junit
#   scripts/pipelines/test.sh --all --cov --html

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

ofb_load_dotenv || true

# ---------- Args ----------
ALL=0
STAGED=0
SINCE=""
KW=""          # -k 'expr'
MARKERS=""     # -m 'expr'
VERBOSE=0
LAST_FAILED=0
FAILED_FIRST=0
WORKERS=""     # auto | N (requires pytest-xdist if set)
COV=1          # coverage on by default
COV_HTML=0
COV_XML=1
JUNIT=0
EXTRA_PYTEST=()  # pass-through
declare -a USER_PATHS=()

usage() {
  cat <<'EOF'
OpenFreqBench tests

Selection:
  --all                Run all tests (default paths)
  --staged             Only tests touching staged files (git)
  --since <ref>        Only tests changed since <ref> (e.g., main)
  --paths <...>        Explicit test paths (rest of args)

Filters:
  -k "<expr>"          Pytest keyword filter
  --markers "<expr>"   Pytest -m marker expression

Behavior:
  -v                   Verbose pytest
  --last-failed        Run only last failed
  --ff                 Failures first

Workers:
  --workers auto|N     Enable pytest-xdist (if installed)

Coverage & reports:
  --no-cov             Disable coverage
  --html               Emit HTML coverage (.ofb/coverage/html)
  --no-xml             Disable XML coverage
  --junit              Emit JUnit XML to .ofb/test-reports/junit.xml

Other:
  --                   Everything after is passed to pytest verbatim
  -h, --help           Show help
EOF
}

while (( "$#" )); do
  case "$1" in
    --all) ALL=1; shift ;;
    --staged) STAGED=1; shift ;;
    --since) SINCE="$2"; shift 2 ;;
    --paths) shift; USER_PATHS=("$@"); break ;;
    -k) KW="$2"; shift 2 ;;
    --markers) MARKERS="$2"; shift 2 ;;
    -v) VERBOSE=1; shift ;;
    --last-failed) LAST_FAILED=1; shift ;;
    --ff) FAILED_FIRST=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --no-cov) COV=0; shift ;;
    --html) COV_HTML=1; shift ;;
    --no-xml) COV_XML=0; shift ;;
    --junit) JUNIT=1; shift ;;
    --) shift; EXTRA_PYTEST+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) warn "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- Target discovery ----------
git_available() { command -v git >/dev/null 2>&1 && [[ -d .git ]]; }

default_paths=( "tests" )
paths=()

if (( ${#USER_PATHS[@]} > 0 )); then
  paths=("${USER_PATHS[@]}")
elif [[ $ALL -eq 1 ]]; then
  paths=("${default_paths[@]}")
elif [[ -n "$SINCE" ]] && git_available; then
  mapfile -t changed < <(git diff --name-only --diff-filter=ACMRT "$SINCE"... -- 'tests/**/*.py' || true)
  paths=("${changed[@]}")
elif [[ $STAGED -eq 1 ]] && git_available; then
  mapfile -t staged < <(git diff --name-only --cached --diff-filter=ACMRT -- 'tests/**/*.py' || true)
  paths=("${staged[@]}")
else
  if git_available; then
    mapfile -t changed < <(git diff --name-only --diff-filter=ACMRT HEAD -- 'tests/**/*.py' || true)
    paths=("${changed[@]}")
    [[ ${#paths[@]} -eq 0 ]] && paths=("${default_paths[@]}")
  else
    paths=("${default_paths[@]}")
  fi
fi

# Filter non-existing
clean_paths=()
for p in "${paths[@]}"; do
  [[ -e "$p" ]] && clean_paths+=("$p")
done
paths=("${clean_paths[@]}")
[[ ${#paths[@]} -gt 0 ]] || { log "No test files to run. Exiting."; exit 0; }

log "Test targets: ${paths[*]}"

# ---------- Reports / logs ----------
REPORT_DIR="$OFB_ROOT/test-reports"
COV_DIR="$OFB_ROOT/coverage"
LOG_DIR="$OFB_ROOT/logs"
mkdir -p "$REPORT_DIR" "$COV_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/test-$(ofb_timestamp).log"
log "Logs: $LOG_FILE"

# ---------- Build pytest args ----------
args=()
(( VERBOSE )) && args+=("-v")
[[ -n "$KW"      ]] && args+=("-k" "$KW")
[[ -n "$MARKERS" ]] && args+=("-m" "$MARKERS")
(( LAST_FAILED )) && args+=("--last-failed")
(( FAILED_FIRST )) && args+=("--ff")

# xdist
if [[ -n "$WORKERS" ]]; then
  # only add -n if xdist is present
  if with_env python -c "import xdist" >/dev/null 2>&1; then
    args+=("-n" "$WORKERS")
  else
    warn "pytest-xdist not installed; ignoring --workers"
  fi
fi

# coverage
if (( COV )); then
  args+=("--cov" ".")
  args+=("--cov-report" "term-missing")
  (( COV_HTML )) && args+=("--cov-report" "html:$COV_DIR/html")
  (( COV_XML  )) && args+=("--cov-report" "xml:$COV_DIR/coverage.xml")
else
  args+=("-p" "no:cov")
fi

# junit
(( JUNIT )) && args+=("--junitxml" "$REPORT_DIR/junit.xml")

# Pass-through
args+=("${EXTRA_PYTEST[@]}")
# Targets last
args+=("${paths[@]}")

# ---------- Run ----------
status=0
( set -o pipefail; with_env pytest "${args[@]}" 2>&1 | tee -a "$LOG_FILE" ) || status=$?

if [[ $status -ne 0 ]]; then
  err "❌ Tests failed (code=$status). See $LOG_FILE"
  exit "$status"
fi

log "✅ Tests passed."
[[ $COV_HTML -eq 1 ]] && log "Coverage HTML: $COV_DIR/html/index.html"
[[ $COV_XML  -eq 1 ]] && log "Coverage XML : $COV_DIR/coverage.xml"
[[ $JUNIT    -eq 1 ]] && log "JUnit XML    : $REPORT_DIR/junit.xml"
