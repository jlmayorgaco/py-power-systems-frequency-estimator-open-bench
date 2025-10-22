#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# OpenFreqBench installer
# - Creates/updates a single conda env
# - Runtime install by default (-e .)
# - With --dev, installs -e ".[dev]" + pre-commit hook
# - Python version selectable via --python
# ------------------------------------------------------------

echo "🚀 Setting up OpenFreqBench environment…"

# ---------- Args ----------
ENV_NAME="openfreqbench"
PY_VER="3.10"
WITH_DEV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) WITH_DEV=1; shift ;;
    --python) PY_VER="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--dev] [--python 3.10|3.11|3.12] [--env NAME]

--dev         Install dev extras (ruff, mypy, pytest, docs, pre-commit)
--python ver  Python version for the env (default: 3.10)
--env name    Conda environment name (default: openfreqbench)
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------- Caches ----------
export OFB_ROOT="${OFB_ROOT:-.ofb}"
export NUMBA_CACHE_DIR="${OFB_ROOT}/cache/numba"
export MPLCONFIGDIR="${OFB_ROOT}/cache/mpl"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# ---------- Pre-flight ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Conda not found. Install Miniconda/Anaconda (or micromamba) and retry." >&2
  exit 1
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- Create / update env ----------
echo "📦 Ensuring conda env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PY_VER}" >/dev/null 2>&1 || true

echo "📦 Base scientific stack (conda-forge)"
conda install -y -n "${ENV_NAME}" -c conda-forge \
  numpy scipy pandas matplotlib numba pywavelets jupyter tqdm pyyaml pip >/dev/null

# ---------- Pip packages ----------
echo "📦 Pip (runtime & optional dev)"
if [[ ${WITH_DEV} -eq 1 ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -U pip wheel
  conda run -n "${ENV_NAME}" python -m pip install -e ".[dev]"
else
  conda run -n "${ENV_NAME}" python -m pip install -U pip wheel
  conda run -n "${ENV_NAME}" python -m pip install -e .
fi

# Optional: extra runtime libs not on conda (keep here if you still need them)
conda run -n "${ENV_NAME}" python -m pip install \
  opendssdirect.py ssqueezepy plotly joblib brokenaxes

# ---------- Dev niceties ----------
if [[ ${WITH_DEV} -eq 1 ]]; then
  # pre-commit hooks (ruff/mypy/pytest can be wired here if you add config)
  if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
    conda run -n "${ENV_NAME}" python -m pip install pre-commit
    conda run -n "${ENV_NAME}" pre-commit install || true
  fi
fi

# ---------- Smoke test ----------
echo "🧪 Verifying imports"
conda run -n "${ENV_NAME}" python - <<'PY'
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

echo "✅ Environment ready!"
echo "👉 Activate: conda activate ${ENV_NAME}"
if [[ ${WITH_DEV} -eq 1 ]]; then
  echo "👉 Run checks: ruff check . && mypy . && pytest && (mkdocs build --strict || sphinx-build …)"
fi
