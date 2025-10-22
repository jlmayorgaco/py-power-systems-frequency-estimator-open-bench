#!/usr/bin/env bash
# OpenFreqBench — docs builder/preview
# Builds MkDocs or Sphinx docs (auto-detect or forced via --stack).
# Examples:
#   scripts/pipelines/docs.sh                 # auto-detect, build
#   scripts/pipelines/docs.sh --serve         # serve mkdocs or sphinx-autobuild
#   scripts/pipelines/docs.sh --stack sphinx --strict
#   scripts/pipelines/docs.sh --linkcheck
#   scripts/pipelines/docs.sh --clean --open

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

# ---------- Defaults & args ----------
STACK=""          # mkdocs|sphinx (auto if empty)
SERVE=0
STRICT=0
CLEAN=0
OPEN=0
LINKCHECK=0
HOST="127.0.0.1"
PORT="8000"

usage() {
  cat <<'EOF'
OpenFreqBench docs

Flags:
  --stack mkdocs|sphinx   Force docs tool (auto-detect if omitted)
  --serve                 Run dev server (mkdocs serve or sphinx-autobuild if present)
  --strict                Fail on warnings (mkdocs --strict / sphinx -W)
  --clean                 Remove previous build output (site/ or docs/_build)
  --open                  Open built docs in browser after success
  --linkcheck             Run Sphinx linkcheck (requires Sphinx)
  --host 127.0.0.1        Host for --serve (default 127.0.0.1)
  --port 8000             Port for --serve (default 8000)
  -h, --help              Show help
EOF
}

while (( "$#" )); do
  case "$1" in
    --stack) STACK="$2"; shift 2;;
    --serve) SERVE=1; shift;;
    --strict) STRICT=1; shift;;
    --clean) CLEAN=1; shift;;
    --open) OPEN=1; shift;;
    --linkcheck) LINKCHECK=1; shift;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

detect_stack() {
  [[ -f "mkdocs.yml" || -f "mkdocs.yaml" ]] && echo "mkdocs" && return
  [[ -f "docs/conf.py" ]] && echo "sphinx" && return
  echo ""
}

STACK="${STACK:-$(detect_stack)}"
[[ -z "$STACK" ]] && die "Could not detect docs stack. Provide mkdocs.yml or docs/conf.py, or use --stack."

LOG_DIR="$OFB_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/docs-$(ofb_timestamp).log"

# ---------- Builders ----------
build_mkdocs() {
  [[ $CLEAN -eq 1 ]] && rm -rf site
  local args=(build)
  [[ $STRICT -eq 1 ]] && args+=(--strict)
  log "MkDocs build → site/"
  ( set -o pipefail; with_env mkdocs "${args[@]}" 2>&1 | tee -a "$LOG_FILE" )
}

serve_mkdocs() {
  local args=(serve -a "${HOST}:${PORT}")
  [[ $STRICT -eq 1 ]] && args+=(--strict)
  log "MkDocs serve → http://${HOST}:${PORT}"
  # Do not tee here (keeps interactive server clean)
  with_env mkdocs "${args[@]}"
}

build_sphinx() {
  [[ $CLEAN -eq 1 ]] && rm -rf docs/_build
  local SPHINXBUILD="sphinx-build"
  local outdir="docs/_build/html"
  local args=(-b html docs "$outdir")
  [[ $STRICT -eq 1 ]] && args=(-W "${args[@]}")
  log "Sphinx build → $outdir"
  ( set -o pipefail; with_env "$SPHINXBUILD" "${args[@]}" 2>&1 | tee -a "$LOG_FILE" )
}

serve_sphinx() {
  local outdir="docs/_build/html"
  if with_env python -c "import sphinx_autobuild" >/dev/null 2>&1; then
    log "sphinx-autobuild → http://${HOST}:${PORT}"
    with_env sphinx-autobuild docs "$outdir" -b html --host "$HOST" --port "$PORT" $([[ $STRICT -eq 1 ]] && echo "-W" || true)
  else
    warn "sphinx-autobuild not installed; falling back to one-off build + simple server."
    build_sphinx
    log "Serving $outdir via Python http.server at http://${HOST}:${PORT}"
    ( cd "$outdir" && with_env python -m http.server "$PORT" --bind "$HOST" )
  fi
}

run_linkcheck() {
  log "Sphinx linkcheck → docs/_build/linkcheck"
  ( set -o pipefail; with_env sphinx-build -b linkcheck docs docs/_build/linkcheck 2>&1 | tee -a "$LOG_FILE" )
}

open_browser() {
  local url
  if [[ "$STACK" == "mkdocs" ]]; then
    url="site/index.html"
  else
    url="docs/_build/html/index.html"
  fi
  if [[ -f "$url" ]]; then
    log "Opening $url"
    with_env python - <<PY
import webbrowser, pathlib
path = pathlib.Path("$url").resolve().as_uri()
webbrowser.open(path)
PY
  else
    warn "Index file not found to open."
  fi
}

# ---------- Dispatch ----------
case "$STACK" in
  mkdocs)
    if [[ $SERVE -eq 1 ]]; then
      serve_mkdocs
    else
      build_mkdocs
      [[ $LINKCHECK -eq 1 ]] && warn "--linkcheck ignored for MkDocs (Sphinx only)."
      [[ $OPEN -eq 1 ]] && open_browser
    fi
    ;;
  sphinx)
    if [[ $SERVE -eq 1 ]]; then
      serve_sphinx
    else
      build_sphinx
      [[ $LINKCHECK -eq 1 ]] && run_linkcheck
      [[ $OPEN -eq 1 ]] && open_browser
    fi
    ;;
  *) die "Unsupported stack: $STACK" ;;
esac

log "✅ Docs task complete. Logs: $LOG_FILE"
