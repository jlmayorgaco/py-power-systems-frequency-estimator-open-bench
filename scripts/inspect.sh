#!/usr/bin/env bash
# OpenFreqBench deep project inspection (macOS-friendly)
# Usage: ./inspect.sh
set -euo pipefail

# ------------------------------- setup ----------------------------------------
REPO_ROOT="$(pwd)"
OUT_DIR=".ofb/inspect"
OUT_FILE="${OUT_DIR}/report.txt"
mkdir -p "$OUT_DIR"

# Colors (fallback if not a TTY)
if [ -t 1 ]; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  CYAN="$(tput setaf 6)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; MAGENTA="$(tput setaf 5)"
else
  BOLD=""; DIM=""; RESET=""; CYAN=""; GREEN=""; YELLOW=""; MAGENTA=""
fi

log() { printf "%s\n" "$*" | tee -a "$OUT_FILE" >/dev/null; }
hr()  { printf "%s\n" "-------------------------------------------------------------------------------" | tee -a "$OUT_FILE" >/dev/null; }
sec() { printf "\n%s\n%s%s%s\n" "$(hr)" "${BOLD}${CYAN}" "$*" "${RESET}" | tee -a "$OUT_FILE" >/dev/null; }
sub() { printf "%s%s%s\n" "${BOLD}${MAGENTA}" "$*" "${RESET}" | tee -a "$OUT_FILE" >/dev/null; }
kv()  { printf "  - %s: %s\n" "$1" "$2" | tee -a "$OUT_FILE" >/dev/null; }

have() { command -v "$1" >/dev/null 2>&1; }

# Begin fresh report
: > "$OUT_FILE"
log "${BOLD}${GREEN}OpenFreqBench — Deep Project Inspection${RESET}"
log "Repo: ${REPO_ROOT}"
log "Date: $(date)"
hr

# ------------------------------- basics ---------------------------------------
sec "BASIC LAYOUT"
if have tree; then
  sub "Top-level tree (2 levels)"
  tree -L 2 -a -I '.git|__pycache__|.mypy_cache|.pytest_cache|.DS_Store|_inspect' | tee -a "$OUT_FILE" >/dev/null
else
  sub "Top-level listing (tree not found; using find)"
  find . -maxdepth 2 -not -path '*/\.*' -not -path './_inspect*' | sort | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- git ------------------------------------------
if [ -d .git ]; then
  sec "GIT SNAPSHOT"
  kv "Current branch" "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
  kv "Last commit"    "$(git log -1 --pretty=format:'%h %ad %an — %s' --date=iso 2>/dev/null || echo 'N/A')"
  sub "Status (short)"
  git status -s | tee -a "$OUT_FILE" >/dev/null || true
  sub "Recent commits (last 10)"
  git --no-pager log --oneline -n 10 | tee -a "$OUT_FILE" >/dev/null || true
fi

# ------------------------------- sizes ----------------------------------------
sec "SIZE & COUNTS"
TOTAL_SIZE=$(du -sh . 2>/dev/null | awk '{print $1}')
PY_COUNT=$(find . -type f -name "*.py" -not -path "./.venv/*" | wc -l | tr -d ' ')
NB_COUNT=$(find notebooks -type f -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
YML_COUNT=$(find . -type f \( -name "*.yml" -o -name "*.yaml" \) | wc -l | tr -d ' ')
TEST_COUNT=$(find tests -type f -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
kv "Total size" "$TOTAL_SIZE"
kv "Python files" "$PY_COUNT"
kv "Notebooks" "$NB_COUNT"
kv "YAML files" "$YML_COUNT"
kv "PyTests" "$TEST_COUNT"

sub "Largest 20 files"
# Portable BSD-compatible sort on mac
find . -type f -not -path "./.git/*" -not -path "./_inspect/*" -not -path "./.venv/*" -exec du -h {} + 2>/dev/null \
| sort -h -r | head -n 20 | tee -a "$OUT_FILE" >/dev/null

# ------------------------------- env ------------------------------------------
sec "ENVIRONMENT.YML SNAPSHOT"
if [ -f environment.yml ]; then
  NAME=$(grep -E '^name:' environment.yml | awk '{print $2}' || echo "N/A")
  PYV=$(grep -E '^- *python(=|:)' environment.yml || true)
  kv "Conda env name" "${NAME:-N/A}"
  kv "Python spec" "${PYV:-unspecified}"
  sub "Top deps (first 30)"
  awk 'NR==1, NR>200{print} NR>200{exit}' environment.yml | sed '1,1!b' | tee -a "$OUT_FILE" >/dev/null
else
  log "${YELLOW}No environment.yml found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- language/loc ---------------------------------
sec "LANGUAGE & LOC ESTIMATE"
if have cloc; then
  sub "cloc summary"
  cloc --exclude-dir=_inspect,.git,.venv --quiet . | tee -a "$OUT_FILE" >/dev/null
else
  sub "cloc not available; rough Python LOC via wc"
  find . -type f -name "*.py" -not -path "./.venv/*" -exec wc -l {} + \
    | sort -n | tail -n 20 | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- estimators -----------------------------------
sec "ESTIMATORS PACKAGE INVENTORY"
if [ -d estimators ]; then
  sub "Python packages/modules under estimators/"
  find estimators -type f -name "*.py" | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null

  sub "Likely estimator classes (regex: class .*Estimator)"
  if have rg; then
    rg --no-heading --line-number --regexp 'class +[A-Za-z0-9_]+.*Estimator' estimators | tee -a "$OUT_FILE" >/dev/null || true
  else
    grep -RIn 'class \+[A-Za-z0-9_]\+.*Estimator' estimators 2>/dev/null | tee -a "$OUT_FILE" >/dev/null || true
    # fallback simpler pattern
    grep -RIn 'class .*Estimator' estimators 2>/dev/null | tee -a "$OUT_FILE" >/dev/null || true
  fi

  sub "Init files"
  find estimators -type f -name "__init__.py" -exec echo "  - {}" \; | tee -a "$OUT_FILE" >/dev/null
else
  log "${YELLOW}estimators/ directory not found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- scenarios ------------------------------------
sec "SCENARIOS INVENTORY"
if [ -d scenarios ]; then
  sub "Scenario modules & YAMLs"
  find scenarios -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" \) \
    | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null

  sub "Mentions of OpenDSS integration"
  if have rg; then
    rg -n "opendss|opendssdirect" scenarios | tee -a "$OUT_FILE" >/dev/null || true
  else
    grep -RIn "opendss\|opendssdirect" scenarios 2>/dev/null | tee -a "$OUT_FILE" >/dev/null || true
  fi
else
  log "${YELLOW}scenarios/ directory not found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- pipelines ------------------------------------
sec "PIPELINES SNAPSHOT"
if [ -d pipelines ]; then
  sub "Pipelines modules"
  find pipelines -type f -name "*.py" | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null

  sub "CLI entrypoints (if any) looking for if __name__ == '__main__'"
  if have rg; then
    rg -n "__main__" pipelines | tee -a "$OUT_FILE" >/dev/null || true
  else
    grep -RIn "__main__" pipelines 2>/dev/null | tee -a "$OUT_FILE" >/dev/null || true
  fi
else
  log "${YELLOW}pipelines/ directory not found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- evaluation -----------------------------------
sec "EVALUATION (metrics, compliance, plots)"
if [ -d evaluation ]; then
  sub "Evaluation modules"
  find evaluation -type f -name "*.py" | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null

  sub "Metrics & compliance keywords"
  if have rg; then
    rg -n "IEC|IEEE|60255|frequency_error|rocof|settling|overshoot" evaluation \
      | tee -a "$OUT_FILE" >/dev/null || true
  else
    grep -RIn "IEC\|IEEE\|60255\|frequency_error\|rocof\|settling\|overshoot" evaluation 2>/dev/null \
      | tee -a "$OUT_FILE" >/dev/null || true
  fi
else
  log "${YELLOW}evaluation/ directory not found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- docs/notebooks -------------------------------
sec "DOCS & NOTEBOOKS"
if [ -d docs ]; then
  sub "Docs tree (1 level)"
  if have tree; then tree -L 1 docs | tee -a "$OUT_FILE" >/dev/null
  else find docs -maxdepth 1 -type f | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null; fi
fi

if [ -d notebooks ]; then
  NB_LIST=$(find notebooks -type f -name "*.ipynb" | wc -l | tr -d ' ')
  kv "Notebooks found" "$NB_LIST"
  sub "Notebook filenames"
  find notebooks -type f -name "*.ipynb" | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- tests ----------------------------------------
sec "TESTS"
if [ -d tests ]; then
  sub "Test files"
  find tests -type f -name "test_*.py" | sed 's#^#  - #' | tee -a "$OUT_FILE" >/dev/null
  if have python && have pytest; then
    sub "pytest collection (dry run)"
    # Do not actually run tests—just collect
    (pytest -q --collect-only || true) 2>&1 | tee -a "$OUT_FILE" >/dev/null
  fi
else
  log "${YELLOW}tests/ directory not found.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- code smells ----------------------------------
sec "QUICK GREP: TODO/FIXME/NOTE"
if have rg; then
  rg -n "TODO|FIXME|HACK|XXX|NOTE" --hidden --glob '!_inspect/**' \
    | tee -a "$OUT_FILE" >/dev/null || true
else
  grep -RIn "TODO\|FIXME\|HACK\|XXX\|NOTE" . 2>/dev/null \
    | grep -v "/\.git/" | grep -v "/_inspect/" | tee -a "$OUT_FILE" >/dev/null || true
fi

# ------------------------------- py introspect --------------------------------
sec "PYTHON INTROSPECTION (best-effort)"
PYBIN="$(command -v python3 || true)"
if [ -n "${PYBIN}" ]; then
  kv "Python" "$(${PYBIN} --version 2>&1)"
  sub "Can import key libs? (best-effort)"
  "${PYBIN}" - <<'PY' 2>&1 | tee -a "$OUT_FILE" >/dev/null || true
import importlib, sys
mods = ["numpy","scipy","pandas","opendssdirect","matplotlib","numba"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  - {m}: OK")
    except Exception as e:
        print(f"  - {m}: FAIL ({e.__class__.__name__}: {e})")
PY
else
  log "${YELLOW}python3 not found in PATH.${RESET}" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- style/lint -----------------------------------
sec "STYLE CHECK HOOKS (presence only)"
if [ -f pyproject.toml ]; then
  kv "pyproject.toml" "present"
  grep -E 'ruff|black|flake8|pylint|mypy|pytest' pyproject.toml || true | tee -a "$OUT_FILE" >/dev/null
elif [ -f setup.cfg ]; then
  kv "setup.cfg" "present"
  grep -E '\[flake8\]|\[tool:pytest\]|\[mypy\]' -n setup.cfg || true | tee -a "$OUT_FILE" >/dev/null
else
  log "  (no pyproject.toml / setup.cfg detected)" | tee -a "$OUT_FILE" >/dev/null
fi

# ------------------------------- licenses -------------------------------------
sec "LICENSE & HEADERS"
if [ -f LICENSE ]; then
  kv "LICENSE" "found"
  head -n 15 LICENSE | sed 's/^/  /' | tee -a "$OUT_FILE" >/dev/null
else
  kv "LICENSE" "missing"
fi

# ------------------------------- summary --------------------------------------
sec "SUMMARY"
kv "Report saved to" "$OUT_FILE"
log "${GREEN}Done.${RESET}"
