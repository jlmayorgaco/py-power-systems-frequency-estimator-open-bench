# OpenFreqBench: Reviewer 2 Uncompromising Audit Report

**Auditor:** Principal Research Software Architect, PMU/PLL Benchmarking Expert
**Date:** March 2026
**Target:** `OpenFreqBench` Core Repository and Artifact Layers

---

## EXECUTIVE VERDICT: 4.5/10 — Mathematically Flawed Masterpiece
*“A beautiful architectural cathedral built on an unstable methodological foundation.”*

OpenFreqBench aspires to be the definitive IEEE C37 compliance testing and discovery engine for dynamic PMU estimation. In its current state, it is structurally majestic but scientifically disjointed. It presents a façade of over 30 advanced estimators and sophisticated statistical survival analyses, yet masks severe methodological flaws at the signal-processing core.

**Is it Q1 ready?** Absolutely not. If submitted today, Reviewer 2 would rip the transient leakage and unvalidated Machine Learning skeletons apart. 
**Is it worth 6 more months?** Yes. The architecture (`EstimatorRegistry`, `ScenarioBase`, Monte Carlo parallelism, Bayesian Tuning) is incredibly modern. If you fix the mathematical pipelines, strip the fake AI modules, and quarantine the transients, it transforms into an undeniably definitive living benchmark.

---

## PART 1 — RECONSTRUCT THE PROJECT HONESTLY

**True Scientific Vision:** Establish a rigorous threshold testing framework under complex dynamic faults (unbalance, harmonics, nightmarish IBR sequences).
**Current Reality:** A sprawling execution shell where 80% of registered estimators are literal `NotImplementedError` stubs or passive pass-throughs. The system successfully executes Monte Carlo pipelines and builds beautiful LaTeX artifact tables, but standard single-phase metrics crash when fed true IEEE three-phase vectors.
  
**Documentation vs Code Mismatch:**
The `legacy_sgsma` paper contributions (`paper_contributions.txt`) claim deep "Behavioral Taxonomies" and "CRLB efficiency Pareto slopes". Yet the actual `results/q1_suite/` execution proves that estimators like `RA_EKF` and `VFF_RLS` fail silently due to massive namespace mismatches (`RAEKF` vs `RA_EKF`), completely bypassing Bayesian sweeps. The docs claim ML supremacy, but the code reveals that NO neural networks (`PI_GRU`, `Transformer_Freq`) are hooked authentically into the `SuiteRunner`.

---

## PART 2 — IMPLEMENTATION COMPLETENESS AUDIT

The `EstimatorRegistry` is severely bloated. It provides "Architecture Theater":
*   `SRF_PLL`, `SOGI_FLL`, `TKEO_DESA2`, `IpDFT`, `TFT`, `RLS`: **REAL_IMPLEMENTED**
*   `RA_EKF`: **BROKEN_WIRING** (Crashes entirely inside `q1_benchmark.yaml` due to key collisions)
*   `PI_GRU`, `EchoState_Reservoir`, `Transformer_Freq`: **FAKE / STUB_OR_EMPTY** (Return `60Hz` statically or raise exceptions).
*   Scenarios (`IBR_Nightmare`, `UnbalancedSag`): **REAL_IMPLEMENTED**

*The brutal truth:* Only ~7 estimators are mathematically robust enough to process pure signal. 

---

## PART 3 — METHODOLOGICAL REVIEW (CRITICAL)

The scientific baseline is broken by fundamental execution flaws.

### 1. Transient/Steady-State Metric Contamination
**Severity: CRITICAL**
**Analysis:** You ran a benchmark on `G1_E1_Pure_60Hz`. `TKEO_DESA2` produced an RMSE of `1.98e-10 Hz` (perfect). The `SRF_PLL` produced an RMSE of `0.14 Hz`. How does a PLL fail to track a perfect 60Hz sine wave? 
*Because the `warm_up_s` threshold is universally hardcoded to ~0.1s.* It takes a PLL filter 0.3s-0.5s to aggressively lock from a generic `0` state. The testing engine is capturing the PLL transient initialization and averaging it into the "Steady State" metric output. **Scientific Claim:** Your tables punishing PLLs and EKFs against "fast" algorithms are completely invalid because you didn't let the filters lock.

### 2. The Batch Multiphasic Blindness fault (IpDFT)
**Severity: CRITICAL**
**Analysis:** In the `src/openfreqbench/estimators/monophasic/...`, `BaseEstimator.step()` safely projects 3-phase into 1-phase using index slicing. However, batch estimators like `IpDFT` override `run()`. The 3-phase array `(1024, 3)` cascades blindly into `np.hanning` and `np.fft.rfft` and throws a broadcasting mismatch. Thus, the entire batch pipeline disintegrates under your new "unbalanced sags" scenario.

### 3. Untested Optuna Data Leakage
**Severity: HIGH**
**Analysis:** `OptunaTuner` dynamically optimizes hyper-parameters (Kp, Ki). If it tunes directly over the target validation scenarios without k-fold isolation, your "superior" method is mechanically over-fitted to the exact validation disturbance vector. The metrics become a test of regression, not adaptability.

---

## PART 4 — ARTIFACT & PLOT AUDIT

*   **`legacy_sgsma/figures_scientific`**: Contains static, manually orchestrated thesis data that is wholly disconnected from the new `SuiteRunner`. The graphs in the PDFs are stunning, but if they were generated using the 0.1s warm-up logic, the conclusions are fundamentally poisoned.
*   **The Checkpoint `.ofb_checkpoint.json`**: Beautiful reproducibility mechanism. Captures `started_at` and hashes perfectly. But when it reports `failed` for `VFF_RLS` and `RA_EKF`, the artifact engine ignores them entirely.

---

## PART 5 — SCORECARD

*   **Scientific Clarity:** 8/10
*   **Benchmark Fairness:** 3/10 (Contamination issues)
*   **Methodology Correctness:** 2/10 (Failed array broadcasting, naive warm-ups)
*   **Architecture Quality:** 8/10 (Amazing abstractions)
*   **Implementation Completeness:** 4/10 (Fake estimators)

**Overall Today:** 4.5/10
**Realistic After 3 Months:** 8.5/10 (Q1 Ready)

---

## PART 6 — FINAL VERDICT

**Most Dangerous Illusion:** That you have a Q1 Benchmark suite ready. You have a Q1 *Shell* mapping over deeply simplistic signal extraction math.
**What to stop doing immediately:** Stop generating LaTeX tables and statistical permutation tests (e.g., Fisher IEC compliance). You cannot measure compliance if your warm-up window is punishing the core filtering mechanism.
**One brutally honest paragraph:** You built an aerospace-grade rocket casing to launch a firework. The OOP models, the parallel `SuiteRunner`, the Optuna hyper-tuning, and the scenario parameterization are masterclasses in Python architecture. Yet, the engine chokes on 3-phase slices and judges recursive filters on their first 100 milliseconds. 

**"Am I wasting my time?"** NO. Fix the single-phase broadcasting bug in batch `run()` methods, set dynamic `settling_time_ms` bounds for metric evaluation, delete or explicitly hide the ML stubs, and synchronize the naming tags inside your YAML. If you do this, your framework will dominate the academic PMU benchmarking space for years.
