# OpenFreqBench — Ph.D. Level & IEEE Q1 Publication Roadmap
> Target: IEEE Transactions on Instrumentation and Measurement / IEEE TPWRD + JOSS
> Branch: `q1-arch-final` → `main` | Updated: 2026-03-22
> Perspective: Exigent Peer-Review Protocol (Reviewer 2 Proofing)

The project has achieved massive architectural milestones. The canonical `BaseEstimator` structure is sound, multiple algorithms from legacy versions are now fully integrated as robust O(1) structures, and statistical reporting is active. 

However, to guarantee acceptance in top-tier journals (Q1), the platform must survive the harshest methodological scrutiny. The following roadmap outlines the rigid path to final publication readiness.

---

## Phase 0: The Current State (Consolidated) ✅

*   **Architectural Core:** `EstimatorRegistry`, `ScenarioBase`, determinism, and CLI orchestration are fully functional.
*   **Methodological SOTA Estimators:** We successfully ported standard and state-of-the-art algorithms: SRF-PLL, SOGI-FLL, RA-EKF, PI-GRU, IpDFT, TKEO, TFT, RLS, VFF-RLS. 
*   **Adversarial Datasets:** `IBR_Nightmare`, `IBR_MultiEvent`, `Chamorro` faults are live and active.
*   **Statistical Backbone:** Non-parametric hypothesis testing (Wilcoxon, Kruskal-Wallis) is implemented.

---

## Phase 1: Metrological Rigor & IEEE Standards Compliance `P0` (Missing for Q1)
*Reviewer Context: "The paper claims compliance with IEEE standards but fails to explicitly model Total Vector Error (TVE) or test Out-of-Band interference."*

### P0-01 Analytical TVE Computation
- **Gap:** Currently computing Frequency Error (FE) and ROCOF Error (RFE). We must compute Total Vector Error (TVE) accurately.
- **Action:** Implement exact TVE metrics capturing amplitude and phase derivation inside `metrics/tve.py`.

### P0-02 Out-of-Band Interference (OOBI) Scenarios
- **Gap:** IEEE C37.118.1a-2014 requires resilience against inter-harmonics near the fundamental frequency.
- **Action:** Create `g2/e21_oobi_interference.py` simulating standard-defined sweeping inter-harmonics.

### P0-03 Explicit Pre-Fault / Post-Fault Signal Segmentation 
- **Gap:** True dynamic metric compliance requires measuring Settling Time *only* after standard fault clearing limits.
- **Action:** Harden event detection masks inside `metrics/frequency.py` for absolute accuracy on transient windows.

---

## Phase 2: Fairness & Hyperparameter Optimization `P1` (Missing for Ph.D. Level)
*Reviewer Context: "How were the hyperparameters chosen for the benchmark? It is unfair to compare a manually-tuned PLL against a default RLS."*

### P1-01 Bayesian Hyperparameter Tuner
- **Gap:** No automated way to find optimal parameters out of the `TuningSpec` defined in each estimator.
- **Action:** Integrate `optuna` into a `tuning/optuna_tuner.py` module.
- **Workflow:** For each method, run 100 Optuna trials over the tuning space on a validation scenario set to find the Pareto optimal config *before* the final benchmark.

---

## Phase 3: Unbalanced 3-Phase Physics `P1`
*Reviewer Context: "Modern grids are rarely perfectly balanced during faults. How do these algorithms perform under severe negative sequence injections?"*

### P1-02 Unbalanced Sag / Swell Scenarios (Types A, B, C, D)
- **Gap:** We have 3-phase matrix ingestions, but no pure mathematically constructed asymmetrical faults.
- **Action:** Implement `g3/e20_unbalanced_sags.py` modeling negative and zero sequence vectors mathematically.

---

## Phase 4: Big-Data Scale & Performance `P2`

### P2-01 SuiteRunner Parallel Execution Engine
- **Gap:** Monte Carlo bounds (1000 seeds) × 15 algorithms × 20 scenarios = 300,000 runs. Python `for` loops will take days.
- **Action:** Refactor `runners/suite_runner.py` to use `concurrent.futures.ProcessPoolExecutor` to parallelize over scenarios/methods seamlessly.

### P2-02 Memory / GC Profiling during Execution
- **Gap:** Python Garbage Collection can cause latency spikes, ruining execution time benchmarking.
- **Action:** Add `gc.disable()` context managers and `tracemalloc` hooks inside `runners/trace_runner.py` to accurately capture algorithmic memory complexity footprint alongside CPU ticks.

---

## Phase 5: Reproducibility & Publication Pack `P2`

### P2-03 Dockerized Environment
- **Action:** Create an immutable `Dockerfile` representing the exact system state that generates the Paper's plots.

### P2-04 CI/CD Verification
- **Action:** Setup `.github/workflows/tests.yml` to run the baseline checks continuously against Python 3.10 to 3.12.

---

## Final Reviewer Summary Matrix
| Metric / Feature | Current State | Target State (Q1 Ready) |
| :--- | :--- | :--- |
| **Statistical Proofs** | Wilcoxon / Kruskal | + Effect Sizes, Confidence Intervals |
| **Testing Regimes** | Nominals, Harmonics, Steps | + OOBI, Unbalanced 3-Phase, TVE |
| **Method Tuning** | Hardcoded Defaults | Optuna Bayesian Optimization |
| **Benchmarking Engine** | Single-threaded | Multi-processing Parallel Engine |
