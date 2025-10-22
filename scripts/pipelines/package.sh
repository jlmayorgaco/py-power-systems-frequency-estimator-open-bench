#!/usr/bin/env bash
# OpenFreqBench — package builder & publisher
# Builds sdist/wheel, performs checks, and optionally uploads to (Test)PyPI.
#
# Examples:
#   scripts/pipelines/package.sh                   # clean build, twine check
#   scripts/pipelines/package.sh --upload testpypi # upload to TestPyPI
#   scripts/pipelines/package.sh --upload pypi --sign
#   scripts/pipelines/package.sh --bump patch      # bump version in pyproject.toml then build
#   scripts/pipelines/package.sh --version 0.1.3   # set exact version then build
#   scripts/pipelines/package.sh --verify-install  # build, then pip install wheel in temp venv
#
# Requirements (declared in pyproject dev extras): build, twine, tomlkit (for bump), wheel.

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
CLEAN=1
CHECK=1
UPLOAD=""            # "", "pypi", or "testpypi"
REPO_URL_PYPI="https://upload.pypi.org/legacy/"
REPO_URL_TEST="https://test.pypi.org/legacy/"
SIGN=0               # GPG sign
VERIFY_INSTALL=0
EXPORT_REQ=0         # export requirements.txt from lock if possible
BUMP_KIND=""         # patch|minor|major|pre|post|dev
SET_VERSION=""       # explicit version X.Y.Z
TAG=0                # create a git tag after bump (vX.Y.Z)
DRY_RUN=0

usage() {
  cat <<'EOF'
OpenFreqBench packager

Flags:
  --no-clean            Do not remove dist/ before build
  --no-check            Skip 'twine check'
  --upload pypi|testpypi  Upload artifacts using twine
  --sign                GPG-sign artifacts (.asc)
  --verify-install      Install the built wheel in a temp venv and import sanity
  --export-req          Export requirements.txt from lock (uv/poetry/pdm if available)
  --bump KIND           Bump version in pyproject.toml (patch|minor|major|pre|post|dev)
  --version X.Y.Z       Set exact version in pyproject.toml
  --tag                 Create a git tag 'v<version>' after bump
  --dry-run             Print actions without executing uploads/tagging
  -h, --help            Show help

Environment:
  TWINE_USERNAME / TWINE_PASSWORD (or keyring) for upload
  ENV_NAME, MANAGER, OFB_ROOT (see scripts/common/env.sh)
EOF
}

while (( "$#" )); do
  case "$1" in
    --no-clean) CLEAN=0; shift ;;
    --no-check) CHECK=0; shift ;;
    --upload)   UPLOAD="$2"; shift 2 ;;
    --sign) SIGN=1; shift ;;
    --verify-install) VERIFY_INSTALL=1; shift ;;
    --export-req) EXPORT_REQ=1; shift ;;
    --bump) BUMP_KIND="$2"; shift 2 ;;
    --version) SET_VERSION="$2"; shift 2 ;;
    --tag) TAG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

LOG_DIR="$OFB_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/package-$(ofb_timestamp).log"
log "Logs: $LOG_FILE"

# ---------- Helpers ----------
run_log() { ( set -o pipefail; "$@" 2>&1 | tee -a "$LOG_FILE" ); }

pyproj_path="pyproject.toml"
[[ -f "$pyproj_path" ]] || die "pyproject.toml not found at repo root."

# ---------- Optional version bump / set ----------
bump_version() {
  local kind="$1"
  log "Bumping version: $kind"
  with_env python - <<PY
import re, sys, tomllib, tomlkit, pathlib
p = pathlib.Path("pyproject.toml")
doc = tomlkit.parse(p.read_text())
ver = doc["project"]["version"]
def bump(v, kind):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(.*)?$", v)
    if not m: raise SystemExit(f"Unsupported version: {v}")
    M,mn,pt,suf = map(int,[m.group(1),m.group(2),m.group(3)])+[None] if False else (int(m.group(1)),int(m.group(2)),int(m.group(3)),m.group(4) or "")
    if kind=="patch": pt+=1; suf=""
    elif kind=="minor": mn+=1; pt=0; suf=""
    elif kind=="major": M+=1; mn=0; pt=0; suf=""
    elif kind=="pre":  suf="-rc.1"
    elif kind=="post": suf=".post1"
    elif kind=="dev":  suf=".dev1"
    else: raise SystemExit(f"Unknown bump kind: {kind}")
    return f"{M}.{mn}.{pt}{suf}"
nv = bump(ver, sys.argv[1])
doc["project"]["version"] = nv
p.write_text(tomlkit.dumps(doc))
print(nv)
PY
}

set_version() {
  local ver="$1"
  log "Setting version: $ver"
  with_env python - <<PY
import tomlkit, pathlib, sys
p = pathlib.Path("pyproject.toml")
doc = tomlkit.parse(p.read_text())
doc["project"]["version"] = sys.argv[1]
p.write_text(tomlkit.dumps(doc))
print(sys.argv[1])
PY
}

NEW_VERSION=""
if [[ -n "$BUMP_KIND" && -n "$SET_VERSION" ]]; then
  die "Use either --bump or --version, not both."
fi
if [[ -n "$BUMP_KIND" ]]; then
  NEW_VERSION="$(bump_version "$BUMP_KIND")"
elif [[ -n "$SET_VERSION" ]]; then
  NEW_VERSION="$(set_version "$SET_VERSION")"
fi
if [[ -n "$NEW_VERSION" ]]; then
  log "New version in pyproject.toml → $NEW_VERSION"
  if [[ $TAG -eq 1 ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      log "[dry-run] Would git commit & tag v$NEW_VERSION"
    else
      git add pyproject.toml
      git commit -m "chore: release v$NEW_VERSION" || true
      git tag "v$NEW_VERSION" -m "Release v$NEW_VERSION"
    fi
  fi
fi

# ---------- Clean ----------
if [[ $CLEAN -eq 1 ]]; then
  log "Cleaning dist/ build/ *.egg-info"
  rm -rf dist build ./*.egg-info
fi

# ---------- Optional export requirements.txt ----------
if [[ $EXPORT_REQ -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    log "Exporting requirements.txt via uv"
    run_log with_env uv export --no-hashes --format requirements-txt > requirements.txt || true
  elif command -v poetry >/dev/null 2>&1; then
    log "Exporting requirements.txt via poetry"
    run_log with_env poetry export -f requirements.txt --output requirements.txt --without-hashes || true
  elif command -v pdm >/dev/null 2>&1; then
    log "Exporting requirements.txt via pdm"
    run_log with_env pdm export -o requirements.txt || true
  else
    warn "No uv/poetry/pdm found; skipping requirements export."
  fi
fi

# ---------- Build sdist & wheel ----------
log "Building package (PEP517)"
run_log with_env python -m build

# ---------- Optional sign artifacts ----------
if [[ $SIGN -eq 1 ]]; then
  if command -v gpg >/dev/null 2>&1; then
    log "Signing artifacts with GPG"
    shopt -s nullglob
    for f in dist/*.{whl,tar.gz}; do
      run_log gpg --detach-sign --armor "$f"
    done
    shopt -u nullglob
  else
    warn "gpg not found; skipping signing."
  fi
fi

# ---------- Twine check ----------
if [[ $CHECK -eq 1 ]]; then
  log "Running twine check"
  run_log with_env twine check dist/*
fi

# ---------- Verify install (throwaway venv) ----------
if [[ $VERIFY_INSTALL -eq 1 ]]; then
  log "Verifying install in a temporary virtualenv"
  TMPVENV="$(mktemp -d)"
  with_env python - <<PY
import sys, venv, subprocess, pathlib
d = pathlib.Path("$TMPVENV")
venv.EnvBuilder(with_pip=True).create(d)
pip = d / ("Scripts/pip.exe" if sys.platform.startswith("win") else "bin/pip")
py  = d / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
# install wheel preferring wheel over sdist
import glob
wheels = sorted(glob.glob("dist/*.whl"))
tgz = sorted(glob.glob("dist/*.tar.gz"))
target = wheels[0] if wheels else tgz[0]
subprocess.check_call([str(pip), "install", target])
subprocess.check_call([str(py), "-c", "import openfreqbench as _; print('import ok:', _. __version__ if hasattr(_, '__version__') else 'unknown')"])
print("✅ verify-install: success")
PY
  rm -rf "$TMPVENV" || true
fi

# ---------- Upload ----------
if [[ -n "$UPLOAD" ]]; then
  case "$UPLOAD" in
    pypi)    REPO="$REPO_URL_PYPI" ;;
    testpypi) REPO="$REPO_URL_TEST" ;;
    *) die "Unknown upload target: $UPLOAD (use pypi|testpypi)" ;;
  esac
  if [[ $DRY_RUN -eq 1 ]]; then
    log "[dry-run] Would upload to $UPLOAD ($REPO)"
  else
    log "Uploading to $UPLOAD ($REPO)"
    # TWINE_USERNAME / TWINE_PASSWORD should be set, or keyring configured.
    run_log with_env twine upload --repository-url "$REPO" dist/*
  fi
fi

log "✅ Packaging complete."
