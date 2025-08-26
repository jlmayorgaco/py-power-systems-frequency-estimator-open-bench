# OpenFreqBench  
*Open Benchmark of Power-System Frequency Estimators*  

---

## Overview  

**OpenFreqBench** is an open, reproducible Python platform for benchmarking frequency and ROCOF estimation algorithms in electric power systems.  
It provides a common testbed where classic, modern, and emerging methods are implemented and evaluated under standardized scenarios.  

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



py-openfreqbench/
├─ estimators/ # All implemented methods
├─ scenarios/ # Synthetic + IEEE systems definitions
├─ pipelines/ # Data generation, benchmarking, summaries
├─ evaluation/ # Metrics, standards compliance, plots
├─ notebooks/ # Examples and tutorials
├─ data/ # Generated datasets and results
└─ docs/ # Documentation




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
conda env create -f environment.yml
conda activate openfreqbench


## Getting Started
from estimators.basic import ipdft
from scenarios.s1_synthetic import make_clean
from evaluation.metrics import frequency_error

# Generate synthetic test signal
signal, truth = make_clean(f0=60.0, df=0.2, duration=5.0, fs=5000)

# Run estimator
est = ipdft.IpDFT(fs=5000, frame_len=256)
f_hat = [est.update(chunk) for chunk in signal]

# Evaluate accuracy
print("RMSE:", frequency_error(f_hat, truth))



## Documentation

Full documentation, examples, and API references are available in the docs/


## Citation
If you use this project in academic work, please cite:

@misc{openfreqbench2025,
  author       = {Mayorga, Jorge Luis},
  title        = {OpenFreqBench: Open Benchmark of Power-System Frequency Estimators},
  year         = {2025},
  url          = {https://github.com/IngJorgeLuisMayorga/py-openfreqbench}
}



## License

This project is licensed under the Apache License 2.0 – see the LICENSE

## Acknowledgements

IEEE PES Task Force benchmark systems (13-bus, 39-bus, 8500-node)

OpenDSS and opendssdirect.py for feeder simulations

Research community contributions on frequency estimation methods

## Contributing

Contributions are welcome. Please open issues or pull requests for:

New estimator implementations
Additional benchmark scenarios
Improvements in metrics and compliance tests
Documentation and tutorials
