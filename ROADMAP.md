# OpenFreqBench — Roadmap
> Target: IEEE Transactions on Power Delivery + JOSS open-source release
> Branch: `q1-arch-final` → `main` | Updated: 2026-03-19

Tickets are ordered by dependency. Each phase has a P-number for triage.
`P0` = blocks everything else. `P1` = blocks publication. `P2` = required for JOSS. `P3` = nice-to-have.

---

## Current State (Phase 0 — Complete ✅)

**232 tests pass. `ofb run` works end-to-end.**

| Deliverable | Status |
|-------------|--------|
| Package scaffold (120+ files) | ✅ |
| `BaseEstimator` + 7 real implementations | ✅ |
| `ScenarioBase` + 3 real scenarios | ✅ |
| `compute_metrics()` — 25+ metric keys | ✅ |
| `aggregate_monte_carlo()` — N seeds → stats | ✅ |
| `TraceRunner` + `ScenarioMethodRunner` | ✅ |
| `ArtifactStore` (deterministic cache) | ✅ |
| `BenchmarkConfig` (Pydantic v2 + YAML) | ✅ |
| `EstimatorRegistry` + `ScenarioRegistry` | ✅ |
| `CheckpointManager` (resume / restart) | ✅ |
| CLI: `ofb run`, `ofb list`, `ofb doctor`, `ofb version` | ✅ |
| 8 estimator stubs (SRF-PLL, RA-EKF, UKF, TFT, RLS, RLS-VFF, Koopman, PI-GRU) | ✅ stubs |

---

## Phase 1 — Estimator Core (target: 15 real estimators) `branch: p1-estimators`

### P0-01 · Port SRF-PLL (real implementation) `P0`
- File: `estimators/monophasic/f0_pll/srf_pll.py`
- Replace stub with full SOGI + dq transform + PI controller
- Parity test vs. `v1/PMU/pfebench/estimators/control/pll_srf.py`
- Required: `update()` returns valid `EstimatorOutput` on 60 Hz sine

### P0-02 · Port RA-EKF (real implementation) `P0`
- File: `estimators/monophasic/f1_kalman/raekf.py`
- Huber M-estimator innovation + Sage-Husa adaptive Q/R
- Parity test vs. `v1/PMU/pfebench/` EKF2 reference
- This is the proposed method for the paper — highest priority

### P0-03 · Port UKF (real implementation) `P0`
- File: `estimators/monophasic/f1_kalman/ukf.py`
- Symmetric unscented transform (α, β, κ parameters)
- Parity test vs. `v1/PMU/pfebench/estimators/states/ukf_freq.py`

### P1-01 · Port RLS (real implementation) `P1`
- File: `estimators/monophasic/f3_recursive/rls.py`
- Phase-unwrap regression window, `delta` regularisation
- Parity test vs. `v1/PMU/pfebench/estimators/regress/rls_phase_unwrap.py`

### P1-02 · Port RLS-VFF (real implementation) `P1`
- File: `estimators/monophasic/f3_recursive/rls_vff.py`
- Variable forgetting factor: λ[n] = λ_min + (1−λ_min)·exp(−|ε|/σ_ref)
- Parity test vs. reference

### P1-03 · Port TFT (real implementation) `P1`
- File: `estimators/monophasic/f2_window/tft.py`
- Taylor-Fourier WLS with polynomial order 1, 2, 3
- Parity test vs. `v1/PMU/pfebench/estimators/poly/taylor_fourier.py`

### P1-04 · Implement `ofb scaffold estimator` CLI command `P1`
- File: `cli/commands/scaffold.py`
- `ofb scaffold estimator MyEst --family f1_kalman`
  → creates `estimators/monophasic/f1_kalman/my_est.py` from Jinja2 template
  → creates `tests/unit/estimators/test_my_est.py`
- Smoke test: generated file imports without error

### P1-05 · Spectral extras (3 estimators) `P1`
- Port `goertzel.py`, `hilbert_freq.py`, `phasor_slope.py`
- Each: real `update()`, latency correct, parity test

---

## Phase 2 — Scenario Suite (target: 10 scenarios) `branch: p2-scenarios`

### P0-04 · G2_E3_PhaseJump scenario `P0`
- File: `scenarios/g2/e3_phase_jump.py`
- Instantaneous phase discontinuity (10°, 30°, 45°)
- `events` list with `EventMarker(type="phase_jump")`
- Unit test: `f_true` unchanged, `phi` has discontinuity

### P0-05 · G1_E2_NoiseSNR scenario `P0`
- File: `scenarios/g1/e2_noise_snr.py`
- AWGN at configurable SNR (20, 30, 40, 60 dB)
- `tuning_map: {"seed": "seed", "snr_db": "snr_db"}`
- MC reproducibility test

### P1-06 · G2_E4_Harmonics scenario `P1`
- File: `scenarios/g2/e4_harmonics.py`
- 3rd and 5th harmonic injection (0–5% THD)
- Unit test: `v_dsp` spectral content confirmed

### P1-07 · G2_E5_Modulation scenario `P1`
- File: `scenarios/g2/e5_modulation.py`
- AM/FM modulation at f_mod = 2 Hz (IEEE C37.118 compliance test)
- Required for IEC compliance column in paper tables

### P1-08 · G3_E1_FreqRampIBR scenario `P1`
- File: `scenarios/g3/e1_freq_ramp_ibr.py`
- Inverter-based resource ramp: fast voltage oscillation during ramp
- IBR scenario — key differentiator from IEC test suite

### P1-09 · G3_E2_IslandingNightmare scenario `P1`
- File: `scenarios/g3/e2_islanding_nightmare.py`
- Combined: freq ramp + phase jump + amplitude sag + harmonics
- This is the "adversarial" scenario — central to paper's main result

### P2-01 · `BENCHMARK_SPEC.md` — Fill physics equations `P2`
- Physics formula for each scenario (analytic signal, noise model, THD)
- Compliance thresholds (IEEE C37.118.1-2011, NERC PRC-024)
- Monte Carlo protocol (seed range, n_runs per scenario type)

---

## Phase 3 — Statistics Engine `branch: p3-stats`

### P0-06 · Bootstrap confidence intervals `P0`
- File: `stats/intervals.py`
- `bootstrap_ci(samples, n_boot=5000, ci=0.95)` → `(lower, upper, point_est)`
- Percentile method (BCa for skewed distributions)
- Known-answer test: CI covers true mean with correct probability

### P0-07 · Pairwise Wilcoxon signed-rank test `P0`
- File: `stats/hypothesis.py`
- `pairwise_wilcoxon(results_dict)` — all pairs over common scenarios
- Bonferroni correction for m*(m-1)/2 comparisons
- Returns `HypothesisResult(statistic, p, p_corrected, reject_H0, effect_size)`
- Required for any significance claim in the paper

### P0-08 · Effect sizes `P0`
- File: `stats/effect_sizes.py`
- `cohens_d(a, b)`, `cliffs_delta(a, b)`, `rank_biserial_r(U, n1, n2)`
- Magnitude labels: negligible/small/medium/large
- Required: IEEE claim "Method A significantly better (d=0.8, p<0.01)"

### P1-10 · Mann-Whitney U + Kruskal-Wallis `P1`
- File: `stats/hypothesis.py` (extend)
- `kruskal_wallis(groups)` — k-way comparison
- `mann_whitney_u(a, b, alternative)` — two-sample, one/two-sided
- Required for pre-registered hypotheses H1–H15 (CLAUDE2.md Block 6)

### P1-11 · Pareto ranking `P1`
- File: `stats/ranking.py`
- `rank_estimators(suite_result, objectives=["RMSE_HZ", "TIME_PER_SAMPLE_US"])`
- Pareto dominance + Borda count fallback
- Returns sorted DataFrame with rank, n_dominated, scores

### P1-12 · Pre-registered hypothesis suite (H1–H15) `P1`
- File: `stats/hypothesis_suite.py` (new)
- 15 pre-registered hypotheses from CLAUDE2.md Block 6
- `run_hypothesis_suite(mc_raw, results_dict)` → full report dict
- Bonferroni AND Benjamini-Hochberg corrections
- All tests include: test_type, statistic, p_value, effect_size, effect_magnitude

### P2-02 · IEC blindness permutation test `P2`
- File: `stats/information.py` (new)
- `iec_blindness_test(results, n_permutations=10000)`
- Core finding: ρ(IEC_RMSE, IBR_RMSE) ≈ 0 → IEC compliance predicts nothing
- Bootstrap CI on Spearman rho

### P2-03 · Cramér-Rao Lower Bound `P2`
- File: `stats/information.py`
- `compute_crlb(scenario_output, fs_dsp)` — theoretical minimum error
- `compute_efficiency(rmse, crlb)` — how close to optimal
- Used in paper Table III footnotes

### P2-04 · Stochastic/deterministic classifier `P2`
- File: `stats/information.py`
- Classify each (method, scenario, metric) by CV:
  - `DETERMINISTIC`: CV < 5%
  - `MODERATE`: 5% ≤ CV < 50%
  - `STOCHASTIC`: CV ≥ 50%
- Key finding: EKF2 Ttrip under islanding is STOCHASTIC (CV ≈ 298%)

---

## Phase 4 — Suite Runner + Parallelism `branch: p4-suite`

### P1-13 · Implement `ScenarioRunner` `P1`
- File: `runners/scenario_runner.py`
- `ScenarioRunner.run(scenario, estimators, cfg)` → `ScenarioResult`
- Integrates checkpoint: skip completed (method, scenario) pairs

### P1-14 · Implement `SuiteRunner` `P1`
- File: `runners/suite_runner.py`
- `SuiteRunner.run(config)` → `BenchmarkResult`
- Loops all scenarios × all estimators, calls `ScenarioRunner`
- Persists each result atomically on completion

### P1-15 · Wire `SuiteRunner` into `ofb run` `P1`
- `cli/commands/run.py` — use `SuiteRunner` instead of nested loops
- Progress: `scenario[3/10] × method[7/15] × seed[42/100]`

### P2-05 · Parallel execution (ProcessPoolExecutor) `P2`
- File: `runners/worker_pool.py`
- `WorkerPool(max_workers=N)` — configurable parallelism
- Config: `benchmark.parallel_workers: 4`
- Crash-safe: each worker writes independently, checkpoint per-pair

### P2-06 · `ofb status` command (full implementation) `P2`
- File: `cli/commands/status.py`
- Scan `artifacts/` + checkpoint → table (scenario × method: seeds_done/total)
- Show estimated time remaining

---

## Phase 5 — Reports & Figures `branch: p5-reports`

### P1-16 · `ofb analyze` command `P1`
- File: `cli/commands/analyze.py`
- Load all JSON artifacts → run stats → write Markdown + CSV summary
- Calls: `ranking.rank_estimators()`, `hypothesis.pairwise_wilcoxon()`
- Output: `results/analysis_YYYYMMDD_HHMMSS.md`

### P1-17 · Scenario comparison table `P1`
- File: `reports/scenario_report.py`
- Per-scenario: all methods, RMSE_HZ (mean ± CI), FE_MAX_MHZ, CPU_US
- Markdown + LaTeX output (`\begin{tabular}`)
- Stars for significance: `*p<0.05`, `**p<0.01`, `***p<0.001`

### P1-18 · Suite heatmap (scenarios × estimators) `P1`
- File: `reports/suite_report.py` + `plotting/suite_plots.py`
- Color = RMSE_HZ (log scale), red=bad/green=good
- Required for paper Fig. 2 (heatmap)

### P1-19 · Pairwise comparison report `P1`
- File: `reports/comparison.py`
- Wilcoxon p-value, Cohen's d, win/tie/loss per pair
- Required for paper supplement Table S1

### P2-07 · Pareto plot `P2`
- File: `plotting/pareto.py`
- RMSE_HZ vs TIME_PER_SAMPLE_US scatter (log-log)
- Points coloured by family, Pareto frontier line
- Required for paper Fig. 3 (efficiency frontier)

### P2-08 · Time-series diagnostic plots `P2`
- File: `plotting/scenario_plots.py` (extend existing)
- `f_hat` vs `f_true` overlay, error signal, event markers
- Publication-ready (LaTeX font, no seaborn)

### P2-09 · IEC compliance heatmap `P2`
- File: `plotting/suite_plots.py` (extend)
- Binary pass/fail per (method, compliance check)
- Separate normative (IEC standard) and non-normative (IBR) columns

---

## Phase 6 — Full Estimator Library (target: 45+) `branch: p6-estimators-ext`

### P2-10 · PLL/Control family (4 estimators) `P2`
- `pll_ddsrf.py`, `epll.py`, `anf.py`, `pr_pll.py`
- Source: `v1/PMU/pfebench/estimators/control/`

### P2-11 · Polynomial family (3 estimators) `P2`
- `taylor_fourier.py` (already in TFT stub), `dynamic_phasor.py`, `ppie.py`

### P2-12 · Regression family (5 estimators) `P2`
- `ls_phase_unwrap.py`, `tls_phase_unwrap.py`, `sg_filter_freq.py`, `wls_rocof.py`
- RLS already in Phase 1; TLS and LS variants here

### P2-13 · Parametric/Subspace family (6 estimators) `P2`
- `esprit.py`, `music.py`, `prony.py`, `matrix_pencil.py`, `nlls_sinefit.py`, `tls_sinefit.py`
- High latency methods; `structural_latency_samples()` must be correct

### P2-14 · State-space family (4 estimators) `P2`
- `kf_freq.py`, `ckf_freq.py`, `pf_freq.py`, `imm_kf.py`
- (EKF and UKF covered in Phase 1)

### P2-15 · Sparse recovery family (4 estimators) `P2`
- `atomic_norm.py`, `lasso_spectrum.py`, `omp_tones.py`, `spice.py`
- `scipy` allowed; add optional dep: `pip install -e ".[sparse]"`

### P2-16 · Time-frequency family (4 estimators) `P2`
- `stft_ridge.py`, `wavelet_if.py`, `sst_cwt.py`, `hht_emd.py`
- `PyWavelets` optional dep: `pip install -e ".[tf]"`

### P2-17 · Hybrid/Ensemble family (5 estimators) `P2`
- `pll_kf_fusion.py`, `dkf_consensus.py`, `ensemble_blend.py`, `rocof_direct.py`
- `ml_cnn_reg.py` requires `torch` → `pip install -e ".[ml]"`

---

## Phase 7 — Advanced Scenarios (target: 18+) `branch: p7-scenarios-ext`

### P2-18 · IEEE 13-bus OpenDSS scenarios (5) `P2`
- `g3/e3_fault_slg_bus671.py`, `g3/e4_motor_start.py`, `g3/e5_pv_ramp.py`
- `g3/e6_reg_tap_step.py`, `g3/e7_unbalance_event.py`
- Requires: `pip install -e ".[opendss]"`

### P2-19 · IEEE 8500-bus scenarios (3) `P2`
- `g3/e8_feeder_faults.py`, `g3/e9_ibr_trip.py`, `g3/e10_pv_cloud.py`

### P2-20 · IEEE 39-bus / Kundur (3) `P2`
- `g3/e11_gen_trip.py`, `g3/e12_governor_step.py`, `g3/e13_two_area_osc.py`

### P3-01 · Real PMU waveform scenarios (2) `P3`
- `g4/e1_lab_pmu.py`, `g4/e2_pmu_event.py`
- CSV loader with f_ref auto-detection; `f_ref_available=False` if no ground truth
- Requires data files (not committed; `data/README.md` with download instructions)

### P3-02 · Chamorro-2021 real data scenario `P3`
- File: `scenarios/data/chamorro_2021.py`
- Auto-detect CSV columns (case-insensitive), normalise voltage to pu
- Ethical note in `meta` dict; relative RMSE if no ground truth

---

## Phase 8 — Publication `branch: p8-paper`

### P1-20 · Full 45×18 benchmark run `P1`
- `ofb run examples/full_suite.template.yaml`
- Verify: no NaN aggregates, all artifacts saved, checkpoint clean
- Expected runtime: ~8 hours single-core, ~2 hours with 4 workers

### P1-21 · Paper figures (Fig 1–5) `P1`
- `ofb analyze --output paper/figures/`
- Fig 1: scenario waveforms gallery (one per group)
- Fig 2: RMSE heatmap (scenarios × estimators)
- Fig 3: Pareto frontier (RMSE vs CPU)
- Fig 4: IEC compliance vs IBR performance scatter (blindness finding)
- Fig 5: Time-series comparison for Islanding scenario

### P1-22 · Write `paper/paper.md` (JOSS format) `P1`
- Summary, Statement of Need, Methods, API, Results, References
- Target: < 1500 words + 4 figures
- Cite: IEEE C37.118.1-2011, IEC 60255-118-1:2018, NERC PRC-024

### P2-21 · JOSS review checklist `P2`
- `CONTRIBUTING.md` finalized (add estimator / add scenario guides)
- `CITATION.cff` with all authors + DOI
- `make test` passes on Python 3.10, 3.11, 3.12, 3.13
- GitHub Actions CI green on all platforms (Linux, macOS, Windows)
- Zenodo DOI for reproducibility archive

### P2-22 · Docker reproducibility check `P2`
- `docker build -t ofb:latest .`
- `docker run ofb:latest ofb run examples/quick_smoke.yaml`
- Results match non-Docker run (bit-for-bit on same seed)

### P3-03 · Performance regression tests `P3`
- `tests/performance/test_cpu_per_sample.py`
- Assert: ZeroCrossing < 1 µs/sample, EKF < 5 µs/sample, FFTPeak < 10 µs/sample
- Run on CI with `--benchmark` flag (optional)

---

## Milestone Summary

| Milestone | Phases | Estimators (real) | Scenarios | Tests | Unblocks |
|-----------|--------|------------------|-----------|-------|---------|
| **M0** ✅ Foundation | 0 | 7 | 3 | 232 | `ofb run` works |
| **M1** First real comparison | 1+2 | 15 | 10 | ~300 | Compare methods end-to-end |
| **M2** Stats complete | 3 | 15 | 10 | ~350 | Bootstrap CI, significance claims |
| **M3** Automated suite | 4 | 15 | 10 | ~380 | One-command 15×10 run |
| **M4** Figures | 5 | 15 | 10 | ~400 | Paper figures draft |
| **M5** Full estimator set | 6 | 45 | 10 | ~500 | IEEE completeness claim |
| **M6** Full scenario set | 7 | 45 | 18 | ~600 | IEEE Transactions scope |
| **M7** 🎯 Submission | 8 | 45 | 18 | ~650 | JOSS + IEEE Transactions |

---

## Ticket Priority Matrix

| ID | Ticket | Priority | Blocks |
|----|--------|----------|--------|
| P0-01 | Port SRF-PLL (real) | **P0** | M1 |
| P0-02 | Port RA-EKF (real) | **P0** | Paper main result |
| P0-03 | Port UKF (real) | **P0** | M1 |
| P0-04 | G2_E3_PhaseJump scenario | **P0** | M1 |
| P0-05 | G1_E2_NoiseSNR scenario | **P0** | M1 |
| P0-06 | Bootstrap CI | **P0** | Any CI in paper |
| P0-07 | Pairwise Wilcoxon test | **P0** | Significance claims |
| P0-08 | Effect sizes | **P0** | IEEE claim formatting |
| P1-01..15 | Phase 1–4 tickets | **P1** | M1–M4 |
| P2-01..22 | Phase 5–8 tickets | **P2** | M5–M7, JOSS |
| P3-01..03 | Advanced / nice-to-have | **P3** | Post-publication |

---

## Open-Source Release Checklist (JOSS Requirements)

- [ ] OSI-approved license (`LICENSE` — MIT)
- [ ] `CITATION.cff` with DOI
- [ ] `CONTRIBUTING.md` — complete (add estimator, add scenario)
- [ ] `CHANGELOG.md` — accurate history
- [ ] `README.md` — one-command install + quick example
- [ ] All tests pass on Python 3.10–3.13 in CI
- [ ] No hardcoded paths or Windows-only code
- [ ] `pip install openfreqbench` works from PyPI (or Zenodo)
- [ ] Notebook tutorials functional (`examples/notebooks/`)
- [ ] `ofb doctor` reports all-green on clean install
- [ ] Docker image builds and smoke test passes
- [ ] Paper figures reproducible from single command
- [ ] Zenodo archive with frozen tag
