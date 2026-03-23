#!/usr/bin/env bash
# init_project.sh
# Scaffold for py-openfreqbench (OpenFreqBench project)

set -e

# Root folder
PROJECT="py-openfreqbench"
mkdir -p $PROJECT
cd $PROJECT

# Core directories
mkdir -p estimators/{basic,regress,param,poly,control,states,tf,sparse,hybrid}
mkdir -p scenarios/{s1_synthetic,s2_ieee13,s3_ieee8500}
mkdir -p pipelines evaluation notebooks docs data/{raw,truth,results} tests docker

# Repo meta
cat > README.md <<'EOF'
# OpenFreqBench  
*Open Benchmark of Power-System Frequency Estimators*  

See full documentation in the [`docs/`](docs/) folder.
EOF

cat > LICENSE <<'EOF'
Apache License 2.0
See full text at: http://www.apache.org/licenses/LICENSE-2.0
EOF

cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.pkl
*.h5
*.parquet

# Conda/venv
.venv/
.env/
env/
*.egg-info/

# Data
data/raw/
data/results/
data/truth/

# Notebooks
.ipynb_checkpoints/

# VSCode/PyCharm
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF

cat > environment.yml <<'EOF'
name: openfreqbench
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - pandas
  - matplotlib
  - pywavelets
  - numba
  - opendssdirect
  - jupyter
  - tqdm
  - pip
  - pip:
      - ssqueezepy
      - plotly
      - joblib
EOF

# Base estimator API
cat > estimators/base.py <<'EOF'
class EstimatorBase:
    """Base class for all frequency estimators."""
    def __init__(self, fs, frame_len, **kwargs):
        self.fs = fs
        self.frame_len = frame_len
        self.params = kwargs

    def reset(self):
        """Reset internal state."""
        pass

    def update(self, x):
        """Update with a chunk of samples."""
        raise NotImplementedError

    def report(self):
        """Return current frequency estimate and diagnostics."""
        return {"f": None, "rocof": None, "theta": None, "latency": None}
EOF

# Example estimator
cat > estimators/basic/ipdft.py <<'EOF'
import numpy as np
from estimators.base import EstimatorBase

class IpDFT(EstimatorBase):
    """Interpolated DFT estimator (simplified)."""

    def update(self, x):
        X = np.fft.fft(x, n=self.frame_len)
        mag = np.abs(X[: self.frame_len // 2])
        k = np.argmax(mag)
        if k == 0 or k == len(mag) - 1:
            return k * self.fs / self.frame_len
        # quadratic interpolation
        alpha = mag[k - 1]
        beta = mag[k]
        gamma = mag[k + 1]
        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        return (k + p) * self.fs / self.frame_len
EOF

# Scenario skeleton
cat > scenarios/s1_synthetic/make_clean.py <<'EOF'
import numpy as np

def make_clean(f0=60.0, df=0.0, duration=5.0, fs=5000):
    """Generate a clean sinusoid with optional frequency ramp."""
    t = np.arange(0, duration, 1/fs)
    f = f0 + df * t / duration
    theta = 2 * np.pi * np.cumsum(f) / fs
    signal = np.sin(theta)
    return signal, f
EOF

# Metrics skeleton
cat > evaluation/metrics.py <<'EOF'
import numpy as np

def frequency_error(f_hat, f_true):
    """RMSE of frequency estimate vs ground truth."""
    f_hat = np.array(f_hat)
    f_true = np.array(f_true[:len(f_hat)])
    return np.sqrt(np.mean((f_hat - f_true)**2))
EOF

# Notebooks placeholder
echo "# OpenFreqBench Example Notebooks" > notebooks/README.md

# Docs placeholder
echo "# OpenFreqBench Documentation" > docs/index.md

# Dockerfile placeholder
cat > docker/Dockerfile <<'EOF'
FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "openfreqbench", "/bin/bash", "-c"]
EOF

# GitHub workflow stub
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF'
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r <(conda list --explicit)
    - name: Run tests
      run: pytest -q
EOF

echo "✅ Project scaffold created in $PROJECT/"
