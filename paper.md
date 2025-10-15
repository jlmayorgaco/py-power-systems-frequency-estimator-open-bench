---
title: "OpenFreqBench: An Open Benchmark of Power-System Frequency Estimators"
tags:
  - Python
  - Power Systems
  - Frequency Estimation
  - ROCOF
  - Benchmarking
  - IEC/IEEE 60255-118-1
authors:
  - name: Jorge Luis Mayorga Taborda
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher, Bogotá, Colombia
    index: 1
date: 2025-10-15
bibliography: paper.bib
---

# Summary

**OpenFreqBench** is an open, reproducible software platform for **benchmarking frequency and rate-of-change-of-frequency (ROCOF) estimation algorithms** in electric power systems.  
It provides a unified testbed that allows researchers, educators, and engineers to implement, compare, and validate estimators under consistent and standardized conditions.

Reliable frequency and ROCOF estimation are critical for low-inertia and converter-dominated grids.  
As distributed energy resources and fast transients increase, estimation algorithms must be evaluated not only for accuracy but also for dynamic response, latency, and compliance with the **IEC/IEEE 60255-118-1** standard.  
However, there is no open, community-maintained benchmark that covers both classical and emerging approaches.  
**OpenFreqBench** fills this gap by offering a transparent, extensible environment where new methods can be evaluated using common datasets, metrics, and compliance envelopes.

# Statement of Need

Research on power-system frequency estimation is scattered across proprietary simulators, closed test cases, or isolated scripts.  
This fragmentation makes it difficult to reproduce published results and to compare new methods on equal footing.  
**OpenFreqBench** was designed to address this reproducibility and comparability problem by:

1. **Providing standardized test scenarios** — from synthetic waveforms to IEEE benchmark feeders (13-bus, 39-bus, 8500-node) simulated via OpenDSS.
2. **Implementing a broad family of estimators** — including spectral (FFT, IpDFT), parametric (Prony, TLS), state-space (KF, EKF, UKF), and control-oriented (PLL/FLL) methods.
3. **Offering reproducible pipelines** — all simulations, metrics, and plots are stored in structured formats (HDF5/Parquet) with full metadata manifests.
4. **Embedding standard compliance tests** — frequency and ROCOF error envelopes consistent with IEC/IEEE 60255-118-1 are evaluated automatically.
5. **Enabling reproducible research and teaching** — through Jupyter notebooks, Conda/Docker environments, and open datasets.

# Implementation and Architecture

OpenFreqBench is written in **Python 3.11** and organized around five modular packages:

- `estimators/`: base classes and algorithm implementations  
- `scenarios/`: signal generators and OpenDSS-based feeders  
- `pipelines/`: reproducible benchmarking workflows and CLI tools  
- `evaluation/`: metric computation, compliance tests, and plotting  
- `docs/` and `notebooks/`: tutorials and examples

Each estimator inherits from a common `EstimatorBase` interface, ensuring consistent input/output handling and latency accounting.  
Benchmark results are saved in self-contained folders containing the signal, ground truth, estimator outputs, and performance metrics.  
Plots follow IEEE 2-column sizing for direct inclusion in publications.

# Validation

Initial validation includes **Zero-Crossing** and **Interpolated DFT (IpDFT)** estimators evaluated on clean, step, and ramp frequency scenarios.  
For each run, the platform computes frequency error (FE), ROCOF error (RFE), and compliance metrics against IEC/IEEE 60255-118-1 steady-state and dynamic envelopes.  
Preliminary results demonstrate reproducibility across operating systems and Python environments.

# Availability and Reuse

The full source code, documentation, and reproducibility environment are available at:

> **https://github.com/IngJorgeLuisMayorga/py-openfreqbench**

The project is licensed under the **Apache License 2.0**.  
Environment setup is automated via `scripts/install.sh`, and a lightweight CLI (`scripts/run.sh` or `ofb run`) provides command-line reproducibility.  
Researchers can extend the benchmark by adding new estimator classes or scenario definitions without modifying the core framework.

# Acknowledgements

The author thanks the **IEEE PES Task Force on Power System Dynamic Performance** for making benchmark systems publicly available, and the maintainers of **OpenDSS** and **OpenDSSDirect.py** for enabling open feeder simulations.  
This work builds on decades of research in signal processing, control, and power systems estimation, and aims to promote open, reproducible comparison across these domains.
