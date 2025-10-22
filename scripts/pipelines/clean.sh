#!/usr/bin/env bash
# OpenFreqBench — cleaner
# Safely remove caches and build artifacts. Defaults are conservative.
# Usage examples:
#   scripts/pipelines/clean.sh
#   scripts/pipelines/clean.sh --deep
#   scripts/pipelines/clean.sh --results
#   ENV_NAME=openfreqbench scripts/pipelines/clean.sh --pycache
#   scripts/pipelines/clean.sh --dry-run --verbose

set -Eeuo pipefail

# ---------- Resolve paths ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# Optional shared helpers
if [[ -f "$ROOT/scripts/common/lib.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/common/lib.sh"
else
  log(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
  warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
  err(){ printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }
  die(){ err "$*"; exit 1; }
fi

# ---------- Defaults & args ----------
DRY_RUN=0
DEEP=0
RESULTS=0
KEEP_LOGS=1
VERBOSE=0
PYCACHE_ENV=0
ENV_NAME="${ENV_NAME:-openfreqbench}"
MANAGER="${MANAGER:-}"   # conda|mamba|micromamba (auto if empty)

OFB_ROOT="${OFB_ROOT:-$ROOT/.ofb}"
export OFB_ROOT
mkdir -p "$OFB_ROOT" >/dev/null 2>&1 || true

usage() {
cat <<'EOF'
OpenFreqBench cleaner

Flags:
  --deep        Remove extra build artifacts (pip wheel metadata, .nox/.tox, etc.)
  --results     Remove data/results/* (keeps logs unless --no-logs)
  --no-logs     When used with --results, also remove results logs
  --pycache     Also purge Python caches inside the project env (site-packages)
  --dry-run     Show what would be removed, without deleting
  --verbose     Print each path as it's processed
  --yes         Do not prompt for confirmation on destructive ops
  -h, --help    Show this help

Env overrides:
  OFB_ROOT        (default: .ofb)
  ENV_NAME        (default: openfreqbench)
  MANAGER         conda|mamba|micromamba (auto-detected if empty)
EOF
}

YES=0
while (( "$#" )); do
  case "$1" in
    --deep) DEEP=1; shift ;;
    --results) RESULTS=1; shift ;;
    --no-logs) KEEP_LOGS=0; shift ;;
    --pycache) PYCACHE_ENV=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# ---------- Manager & env wrapper (only if --pycache) ----------
pick_manager() {
  if [[ -n "$MANAGER" ]]; then echo "$MANAGER"; return; fi
  command -v micromamba >/dev/null && { echo micromamba; return; }
  command -v mamba      >/dev/null && { echo mamba; return; }
  command -v conda      >/dev/null && { echo conda; return; }
  echo ""
}
MANAGER="$(pick_manager)"

with_env() {
  case "$MANAGER" in
    micromamba) micromamba run -n "$ENV_NAME" "$@";;
    mamba)      mamba      run -n "$ENV_NAME" "$@";;
    conda)      conda      run -n "$ENV_NAME" "$@";;
    "")         "$@";;  # fallback to host if manager not available
    *)          die "Unsupported manager: $MANAGER";;
  esac
}

# ---------- Helpers ----------
rm_path() {
  local p="$1"
  [[ -e "$p" || -L "$p" ]] || return 0
  [[ $VERBOSE -eq 1 ]] && echo "  rm -rf $p"
  if [[ $DRY_RUN -eq 0 ]]; then
    rm -rf -- "$p"
  fi
}

confirm() {
  local msg="$1"
  [[ $YES -eq 1 ]] && return 0
  read -r -p "$msg [y/N] " ans || true
  [[ "$ans" =~ ^[Yy]$ ]]
}

# ---------- Target sets ----------
CACHE_DIRS=(
  ".ruff_cache"
  ".mypy_cache"
  ".pytest_cache"
  ".coverage"
  "htmlcov"
  ".ipynb_checkpoints"
  "$OFB_ROOT/cache"
)

BUILD_DIRS=(
  "build"
  "dist"
  "site"
  "*.egg-info"
  "pip-wheel-metadata"
)

DEEP_DIRS=(
  ".nox"
  ".tox"
  ".venv"
  ".vscode/.ropeproject"
)

PY_CACHE_GLOBS=(
  "**/__pycache__"
  "**/*.pyc"
  "**/*.pyo"
)

RESULTS_DIRS=(
  "data/tmp"
  "data/intermediate"
  "data/results"
)

RESULTS_KEEP_LOGS_GLOB="data/results/*/logs"

# ---------- Do the work ----------
log "Cleaning caches and build artifacts"
[[ $DRY_RUN -eq 1 ]] && warn "DRY-RUN: no files will be deleted"

# Caches
for d in "${CACHE_DIRS[@]}"; do
  rm_path "$d"
done

# Python cache files within repo
for g in "${PY_CACHE_GLOBS[@]}"; do
  # find respects globstar via shopt if enabled; use find to be portable
  while IFS= read -r p; do rm_path "$p"; done < <(find . -path "./.git" -prune -o -type d -name "__pycache__" -print)
  while IFS= read -r p; do rm_path "$p"; done < <(find . -type f -name "*.py[co]" -print)
  break  # handled by find; break after first iteration
done

# Build outputs
for d in "${BUILD_DIRS[@]}"; do
  # Expand globs safely
  shopt -s nullglob || true
  for p in $d; do rm_path "$p"; done
  shopt -u nullglob || true
done

# Deep extras
if [[ $DEEP -eq 1 ]]; then
  log "Deep clean enabled"
  for d in "${DEEP_DIRS[@]}"; do
    shopt -s nullglob || true
    for p in $d; do rm_path "$p"; done
    shopt -u nullglob || true
  done
fi

# Results
if [[ $RESULTS -eq 1 ]]; then
  if confirm "This will remove data results and intermediates. Continue?"; then
    for d in "${RESULTS_DIRS[@]}"; do
      if [[ "$d" == "data/results" && $KEEP_LOGS -eq 1 ]]; then
        # keep any logs folders
        shopt -s nullglob || true
        for p in data/results/*; do
          [[ -d "$p" ]] || continue
          if [[ -d "$p/logs" ]]; then
            if [[ $VERBOSE -eq 1 ]]; then echo "  preserving $p/logs"; fi
            tmp_preserve="$(mktemp -d)"
            mv "$p/logs" "$tmp_preserve/" 2>/dev/null || true
            rm_path "$p"
            mkdir -p "$p"
            mv "$tmp_preserve/logs" "$p/" 2>/dev/null || true
            rmdir "$tmp_preserve" 2>/dev/null || true
          else
            rm_path "$p"
          fi
        done
        shopt -u nullglob || true
      else
        rm_path "$d"
      fi
    done
  else
    warn "Skipped results cleanup."
  fi
fi

# Optional: purge env site-packages pyc (rarely needed)
if [[ $PYCACHE_ENV -eq 1 ]]; then
  if [[ -n "$MANAGER" || -n "$(pick_manager)" ]]; then
    log "Purging Python caches inside env: $ENV_NAME"
    with_env python - <<'PY'
import compileall, sys, site, pathlib
paths = [site.getsitepackages()[0], site.getusersitepackages()]
for p in paths:
    try:
        compileall.compile_dir(p, force=True, quiet=1, optimize=0)
    except Exception:
        pass
    for pyc in pathlib.Path(p).rglob("*.py[co]"):
        try: pyc.unlink()
        except Exception: pass
    for d in pathlib.Path(p).rglob("__pycache__"):
        try: d.rmdir()
        except Exception: pass
print("  ✅ env pyc purge attempted.")
PY
  else
    warn "No conda-like manager found; cannot purge env pycache."
  fi
fi

log "✅ Clean complete."
