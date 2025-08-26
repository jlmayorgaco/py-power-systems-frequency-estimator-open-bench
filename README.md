## **OpenFreqBench**

*An Open, Reproducible Python Platform for Benchmarking Power-System Frequency Estimators* ⚡

This README provides a comprehensive overview of **OpenFreqBench**, a platform designed for the standardized benchmarking of frequency and ROCOF (Rate of Change of Frequency) estimation algorithms in electric power systems. It offers a common, reproducible testbed for evaluating classic, modern, and emerging methods under a variety of standardized scenarios.

Built for **research, teaching, and industrial applications**, the project adheres to established **IEEE/IEC standards**, particularly **IEC/IEEE 60255-118-1**, to ensure the relevance and reliability of its results. All code, scenarios, and evaluation metrics are open, transparent, and fully reproducible.

## **Key Features**

OpenFreqBench provides a robust suite of tools for comprehensive analysis.

### **1. Comprehensive Estimator Library**

The platform features a wide array of implemented frequency estimation methods, enabling direct comparisons across diverse algorithmic families.

  * **Classic Algorithms:** Zero-Crossing, Fast Fourier Transform (FFT), Interpolated DFT (IpDFT), and Recursive DFT.
  * **Optimization-based Methods:** Least Squares (LS), Recursive LS (RLS), Total LS (TLS), and Maximum Likelihood (ML) / Non-Linear LS (NLLS).
  * **Parametric Signal Models:** Prony, Matrix Pencil, MUltiple SIgnal Classification (MUSIC), and Estimation of Signal Parameters via Rotational Invariance Techniques (ESPRIT).
  * **Dynamic and Time-Domain Approaches:** Taylor–Fourier and Dynamic Phasor methods.
  * **Phase-Locked Loops (PLLs) and Filters:** The PLL/FLL family, including Synchronous Reference Frame (SRF), Decoupled Double SRF (DDSRF), Second-Order Generalized Integrator (SOGI), Extended PLL (EPLL), and Adaptive Notch Filter (ANF).
  * **State-Space Filters:** Kalman Filter (KF), Extended KF (EKF), Unscented KF (UKF), Cubature KF (CKF), Particle Filter (PF), and Interacting Multiple Model (IMM).
  * **Time-Frequency Analysis:** Hilbert Transform, Short-Time Fourier Transform (STFT), Wavelets, and Synchrosqueezed Transform (SST).
  * **Hybrid & Machine Learning:** A growing collection of methods combining different techniques or leveraging ML approaches.

### **2. Standardized Simulation Scenarios**

To ensure fair comparisons, OpenFreqBench includes a wide range of predefined scenarios, from simple synthetic signals to complex power grid simulations.

  * **Synthetic Signals:** Ideal signals with controlled variations, including noise, harmonics, and dynamic events like steps, ramps, and chirps.
  * **Distribution Feeders:** Detailed simulations of **IEEE 13-bus** and other feeders using **OpenDSS** to model realistic conditions such as unbalance, tap changes, faults, and harmonics.
  * **Transmission Systems:** Analysis of widely recognized test systems, including the **IEEE 39-bus** and **Kundur two-area systems**, to study events like nadir, ROCOF, and inter-area oscillations.
  * **Large-Scale Grids:** Simulations of the **IEEE 8500-node** system with high renewable energy and Inverter-Based Resource (IBR) penetration to analyze low inertia and fast dynamics.

### **3. Robust Evaluation Metrics**

The platform provides a comprehensive suite of metrics to assess estimator performance from multiple angles.

  * **Accuracy:** Quantifies performance using metrics like **Frequency Error (FE)** and **ROCOF Error (RFE)**.
  * **Dynamic Response:** Measures transient behavior with metrics such as rise time, settling time, and overshoot.
  * **Standard Compliance:** Compares performance against compliance envelopes defined by **IEC/IEEE 60255-118-1** to assess grid code adherence.
  * **Computational Efficiency:** Profiles the computational cost and latency of each algorithm, a critical factor for real-time applications.

### **4. Ensuring Reproducibility**

OpenFreqBench is built on a foundation of reproducibility, making it easy for researchers to replicate results and build upon existing work.

  * **OpenDSS Integration:** Seamlessly interacts with OpenDSS via `opendssdirect.py` for power system simulations.
  * **Configurable Scenarios:** Scenarios are defined using **YAML** specifications, allowing for easy modification and version control.
  * **Structured Data Storage:** Results and datasets are stored in organized formats such as **HDF5** and **Parquet**, ensuring data integrity and accessibility.
  * **Reproducible Environment:** A complete environment is provided using **Conda** and **Docker**, eliminating dependency issues.

-----

## **Repository Structure**

The project is organized into clear, functional directories to facilitate navigation and contributions.

```
py-openfreqbench/
├─ estimators/         # All implemented frequency estimation methods
├─ scenarios/          # Definitions for synthetic and IEEE power system scenarios
├─ pipelines/          # Scripts for data generation, benchmarking, and result summaries
├─ evaluation/         # Modules for metrics, standards compliance, and plotting
├─ notebooks/          # Jupyter notebooks with examples and tutorials
├─ data/               # Default location for generated datasets and results
└─ docs/               # Comprehensive project documentation
```

-----

## **Getting Started**

### **Requirements**

  * **Python 3.10+**
  * **Recommended:** Anaconda or Miniconda for environment management.

### **Installation**

Clone the repository and set up the environment.

```bash
git clone https://github.com/IngJorgeLuisMayorga/py-openfreqbench.git
cd py-openfreqbench
conda env create -f environment.yml
conda activate openfreqbench
```

### **Example: Running an Estimator**

The following code snippet demonstrates how to use OpenFreqBench to test a simple estimator.

```python
from estimators.basic import ipdft
from scenarios.s1_synthetic import make_clean
from evaluation.metrics import frequency_error

# 1. Generate a synthetic test signal
signal, truth = make_clean(f0=60.0, df=0.2, duration=5.0, fs=5000)

# 2. Initialize and run the estimator
estimator = ipdft.IpDFT(fs=5000, frame_len=256)
frequency_estimate = [estimator.update(chunk) for chunk in signal]

# 3. Evaluate the accuracy of the result
print("RMSE:", frequency_error(frequency_estimate, truth))
```

-----

## **Documentation**

For detailed information, including API references, in-depth tutorials, and more examples, please refer to the documentation in the **`docs/`** directory.

-----

## **Citation**

If you use this platform in your academic work, please cite it using the following BibTeX entry:

```
@misc{openfreqbench2025,
  author = {Mayorga, Jorge Luis},
  title = {OpenFreqBench: Open Benchmark of Power-System Frequency Estimators},
  year = {2025},
  url = {https://github.com/IngJorgeLuisMayorga/py-openfreqbench}
}
```

-----

## **License**

This project is licensed under the **Apache License 2.0**. Please refer to the **`LICENSE`** file for full details.

-----

## **Acknowledgements**

We extend our gratitude to the following for their foundational contributions:

  * The **IEEE PES Task Force** for providing the benchmark systems (13-bus, 39-bus, 8500-node) that make this research possible.
  * The developers of **OpenDSS** and `opendssdirect.py` for enabling realistic power system simulations.
  * The broader **research community** whose work on frequency estimation methods forms the core of this project.

-----

## **Contributing** 🤝

We welcome and appreciate all contributions to make this platform even better. If you are interested in contributing, please consider the following areas and open an issue or pull request:

  * **New Estimator Implementations:** Add new frequency or ROCOF estimation methods to the `estimators/` library.
  * **Additional Benchmark Scenarios:** Develop new scenarios for the `scenarios/` directory, especially those that model unique grid conditions.
  * **Improvements:** Enhance existing metrics, add new compliance tests, or optimize the codebase.
  * **Documentation:** Improve tutorials, write new examples, or clarify existing documentation.
