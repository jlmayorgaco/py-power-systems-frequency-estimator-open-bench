<!-- Logo / Title -->
<p align="center">
  <img src="docs/assets/openfreqbench_logo.svg" alt="OpenFreqBench logo" width="180">
</p>

<h1 align="center">OpenFreqBench</h1>
<p align="center"><i>Open Benchmark of Power-System Frequency Estimators</i></p>

<!-- Badges (single row, no duplicates) -->
<p align="center">
  <a href="https://github.com/IngJorgeLuisMayorga/py-openfreqbench/actions/workflows/tests.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/IngJorgeLuisMayorga/py-openfreqbench/tests.yml?label=CI&logo=github">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://zenodo.org/doi/TBD"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg"></a>
  <img alt="OS" src="https://img.shields.io/badge/OS-macOS%20%7C%20Linux-lightgrey">
  <!-- Optional when you submit to JOSS:
  <a href="https://joss.theoj.org/papers/TBD"><img alt="JOSS" src="https://joss.theoj.org/papers/TBD/status.svg"></a>
  -->
</p>

<!-- Quick Nav -->
<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="docs/">Docs</a> •
  <a href="#-getting-started">Install</a> •
  <a href="#-quick-example">Run</a> •
  <a href="#-citation">Cite</a>
</p>

<!-- (Optional) hero/diagram; comment out if not ready -->
<p align="center">
  <img src="docs/assets/architecture_diagram.svg" alt="Architecture overview" width="760">
</p>

---

## Overview  

**OpenFreqBench** is an open, reproducible platform for benchmarking **frequency and ROCOF estimators** in electric power systems.  
It provides a **common testbed** where classic, modern, and emerging algorithms are implemented and evaluated under **standardized, IEEE/IEC-aligned scenarios**.

> **Goal:** enable transparent, quantitative comparison of frequency-estimation methods for research, teaching, and industrial applications.


The project is designed for research, teaching, and industrial applications, and aligns with **IEEE/IEC standards** (e.g., IEC/IEEE 60255-118-1).  
All code, scenarios, and results are open and reproducible.  

---

## Key Features  

- **Comprehensive estimator library**  
  - Zero-Crossing, FFT, IpDFT, Recursive DFT  
  - LS, RLS, TLS, ML/NLLS  
  - Prony, Matrix Pencil, MUSIC, ESPRIT  
  - Taylor–Fourier, Dynamic Phasor methods  
  - PLL/FLL family (SRF, DDSRF, SOGI, EPLL, ANF)  
  - State-space filters (KF, EKF, UKF, CKF, PF, IMM)  
  - Time-frequency (Hilbert, STFT, Wavelets, SST)  
  - Hybrid and Machine Learning approaches  

- **Simulation scenarios**  
  - Synthetic signals: clean, noisy, harmonics, steps, ramps, chirps  
  - IEEE 13-bus and other feeders via OpenDSS (unbalance, taps, faults, harmonics)  
  - IEEE 39-bus and Kundur two-area systems (nadir, ROCOF, inter-area modes)  
  - Large-scale IEEE 8500-node with renewable/IBR penetration (low inertia, fast dynamics)  

- **Evaluation metrics**  
  - Frequency Error (FE), ROCOF Error (RFE)  
  - Dynamic response (rise time, settling, overshoot)  
  - Compliance envelopes aligned with **IEC/IEEE 60255-118-1**  
  - Computational cost and latency profiling  

- **Reproducibility**  
  - OpenDSS integration through `opendssdirect.py`  
  - Configurable scenarios (YAML specs)  
  - Results stored in structured formats (HDF5/Parquet)  
  - Full environment provided (Conda + Docker)  

---

## Repository Structure  

```
py-openfreqbench/
├─ estimators/        # Implemented estimators (ZC, IpDFT, KF, PLL, ...)
├─ scenarios/         # Synthetic + IEEE feeder definitions
├─ pipelines/         # Data generation, benchmarking, summaries
├─ evaluation/        # Metrics, compliance, plotting utilities
├─ notebooks/         # Tutorials and reproducible examples
├─ data/              # Generated results (ignored by Git)
├─ docs/              # Documentation and figures
└─ scripts/           # Install / run / clean utilities
```

---

## Getting Started  

### Requirements  
- Python 3.10+  
- Recommended: Anaconda or Miniconda  

### Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/IngJorgeLuisMayorga/py-openfreqbench.git
cd py-openfreqbench
bash scripts/install.sh
conda activate openfreqbench
```

### Quick Example  

```python
# Add Estimator
from estimators.basic import ipdft
# Add Scenarios
from scenarios.s1_synthetic import make_clean
# Add Metrics 
from evaluation.metrics import frequency_error

# Generate synthetic test signal
signal, truth = make_clean(f0=60.0, df=0.2, duration=5.0, fs=5000)

# Run estimator
est = ipdft.IpDFT(fs=5000, frame_len=256)
f_hat = [est.update(chunk) for chunk in signal]

# Evaluate accuracy
print("RMSE:", frequency_error(f_hat, truth))
```


---
## 🧭 Roadmap

| Stage | Feature | Status |
|:------|:---------|:------:|
| Core architecture & packaging | `pyproject.toml`, CLI, base estimator | ✅ |
| Synthetic scenarios (steps, ramps, chirps) | Basic generators | ✅ |
| Evaluation metrics | FE, RFE, RMSE, latency | ✅ |
| IEC/IEEE compliance envelopes | M-class & P-class | 🧩 *in progress* |
| OpenDSS integration (13-bus, 39-bus) | Scenario adapters | 🧩 *in progress* |
| Advanced estimators (KF, PLL, ML) | Library extension | 🚧 *planned* |
| Continuous integration (CI) | GitHub Actions + tests | 🚧 *planned* |
| Paper & citation DOI | Zenodo + JOSS submission | 🚧 *planned* |

---

## 📁 Reproducibility & Results Layout

```
data/results/<timestamp>_<scenario>_<estimator>/
│
├── manifest.json        # run metadata (env, seeds, configs)
├── metrics.parquet      # per-sample FE/RFE
├── summary.json         # RMSE, rise/settle, compliance %
├── plots/               # Figures (FE/RFE vs time, envelopes)
└── logs/                # Pipeline logs
```

---

## Documentation  

Full documentation, examples, and API references are available in the [`docs/`](docs/) folder.  
Notebooks in [`notebooks/`](notebooks/) demonstrate usage with synthetic signals and IEEE test systems.  

---

## Citation  

If you use this project in academic work, please cite:  

```bibtex
@misc{openfreqbench2025,
  author       = {Mayorga, Jorge Luis},
  title        = {OpenFreqBench: Open Benchmark of Power-System Frequency Estimators},
  year         = {2025},
  url          = {https://github.com/IngJorgeLuisMayorga/py-openfreqbench}
}
```

---

## License  

This project is licensed under the **Apache License 2.0** – see the [LICENSE](LICENSE) file for details.  

---

## Acknowledgements  

- IEEE PES Task Force benchmark systems (13-bus, 39-bus, 8500-node)  
- OpenDSS and `opendssdirect.py` for feeder simulations  
- Research community contributions on frequency estimation methods  

---

## Contributing  

Contributions are welcome. Please open issues or pull requests for:  
- New estimator implementations  
- Additional benchmark scenarios  
- Improvements in metrics and compliance tests  
- Documentation and tutorials  
See CONTRIBUTING.md for guidelines and open issues.





<p align="center"><i>Developed and maintained with ⚙️ & ❤️ by Jorge Luis Mayorga Taborda</i></p>