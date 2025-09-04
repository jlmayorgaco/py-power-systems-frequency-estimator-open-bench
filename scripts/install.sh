#!/usr/bin/env bash
set -e  # stop on error

echo "🚀 Setting up OpenFreqBench environment..."

# -----------------------------
# 1. Handle SSL issues globally
# -----------------------------
# Force conda and pip to ignore SSL
export CONDA_SSL_VERIFY=false
export PIP_NO_VERIFY_CERTS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------
# 2. Create / update environment
# -----------------------------
ENV_NAME="openfreqbench"

echo "📦 Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10 || true

echo "📦 Activating environment"
# NOTE: in scripts, we use conda run instead of activate
conda run -n $ENV_NAME python --version

# -----------------------------
# 3. Install conda dependencies
# -----------------------------
echo "📦 Installing conda packages"
conda install -y -n $ENV_NAME -c conda-forge numpy scipy matplotlib pandas pyyaml jupyter

# -----------------------------
# 4. Install pip-only packages
# -----------------------------
echo "📦 Installing pip-only packages"
conda run -n $ENV_NAME pip install opendssdirect

# -----------------------------
# 5. Dev mode install (if you want editable imports)
# -----------------------------
echo "📦 Installing project in editable mode"
conda run -n $ENV_NAME pip install -e .

# -----------------------------
# 6. Done
# -----------------------------
echo "✅ Environment ready!"
echo "To activate it manually, run:"
echo "   conda activate $ENV_NAME"
