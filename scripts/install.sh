#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up OpenFreqBench environment..."


# Centralize caches
export OFB_ROOT="${OFB_ROOT:-.ofb}"
export NUMBA_CACHE_DIR="${OFB_ROOT}/cache/numba"
export MPLCONFIGDIR="${OFB_ROOT}/cache/mpl"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# -----------------------------
# 0) Pre-flight
# -----------------------------
command -v conda >/dev/null 2>&1 || {
  echo "❌ Conda not found. Install Miniconda/Anaconda or use micromamba." >&2
  exit 1
}

ENV_NAME="openfreqbench"
PY_VER="3.11"

# -----------------------------
# 1) SSL tweaks
# -----------------------------
# Keep these OFF by default; uncomment only if you hit corporate MITM/SSL issues.
export CONDA_SSL_VERIFY=false
export PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------
# 2) Create / update environment
# -----------------------------
echo "📦 Creating conda environment: $ENV_NAME (python=$PY_VER)"
conda create -y -n "$ENV_NAME" -c conda-forge "python=${PY_VER}" || true

echo "📦 Python version in env"
conda run -n "$ENV_NAME" python --version

# -----------------------------
# 3) Install conda dependencies
# -----------------------------
echo "📦 Installing conda packages (core scientific stack)"
conda install -y -n "$ENV_NAME" -c conda-forge \
  numpy scipy pandas matplotlib numba pywavelets jupyter tqdm pyyaml pip

# -----------------------------
# 4) Install pip-only packages
# -----------------------------
echo "📦 Installing pip-only packages"
conda run -n "$ENV_NAME" python -m pip install \
  opendssdirect.py ssqueezepy plotly joblib brokenaxes

# -----------------------------
# 5) Dev mode install (only if metadata exists)
# -----------------------------
if [ -f "pyproject.toml" ] || [ -f "setup.cfg" ] || [ -f "setup.py" ]; then
  echo "📦 Installing project in editable mode (-e .)"
  conda run -n "$ENV_NAME" python -m pip install -e .
else
  echo "ℹ️ No packaging metadata (pyproject.toml/setup.*) found; skipping editable install."
  echo "   You can still run with: PYTHONPATH=. python -m pipelines.run_benchmark ..."
fi

# -----------------------------
# 6) Quick import smoke-test
# -----------------------------
echo "🧪 Verifying imports"
conda run -n "$ENV_NAME" python - <<'PY'
mods = ["numpy","scipy","pandas","matplotlib","numba","pywt","tqdm","opendssdirect"]
ok = True
for m in mods:
    try:
        __import__(m)
        print(f"  ✅ {m}")
    except Exception as e:
        ok = False
        print(f"  ❌ {m} -> {type(e).__name__}: {e}")
raise SystemExit(0 if ok else 1)
PY

# -----------------------------
# 7) Done
# -----------------------------
echo "✅ Environment ready!"
echo "👉 To activate it manually:"
echo "   conda activate ${ENV_NAME}"
