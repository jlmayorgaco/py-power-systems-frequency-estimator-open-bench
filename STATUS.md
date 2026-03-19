# OpenFreqBench — Project Status
> Date: 2026-03-19 | Branch: `q1-arch-final` | Version: `0.2.0-dev` | Tests: 292 passing

---

## Quick Start (works today)

```bash
pip install -e ".[dev]"
ofb version                          # → openfreqbench 0.2.0-dev
ofb doctor                           # → all imports green
ofb list                             # → 15 estimators, 3 scenarios
ofb run examples/quick_smoke.yaml    # 5 seeds × G1_E1_Pure_60Hz × ZeroCrossing
ofb run examples/g1_baseline.yaml    # 100 seeds × G1 × 2 estimators
ofb run config.yaml --status         # show checkpoint progress
ofb run config.yaml --resume         # skip completed pairs
pytest tests/                        # 232 pass, 0 fail
```

---

## End-to-End Pipeline (what runs today)

```
ofb run examples/quick_smoke.yaml
  │
  ├─ CheckpointManager — create/resume session
  ├─ BenchmarkConfig.load(yaml)
  │
  └─ for each (scenario × estimator):
       ├─ checkpoint.should_run() → skip if already done
       ├─ ScenarioRegistry.build("G1_E1_Pure_60Hz")
       ├─ EstimatorRegistry.build("ZeroCrossing", params={})
       │
       └─ ScenarioMethodRunner.run(n=5):
            ├─ seed 0..4: TraceRunner.run()
            │   ├─ scenario.set_montecarlo_tuning({"seed": N})
            │   ├─ waveform = scenario.build()       # v, f_true, t numpy arrays
            │   ├─ f_hat, dt_s = TimingHarness.timed_run(estimator, v)
            │   └─ metrics = compute_metrics(...)    # 25+ metric keys
            └─ aggregate_monte_carlo(traces)         # mean/std/p95/p99/n
                 │
                 ├─ ArtifactStore.save_json(...)     # artifacts/.../report.json
                 ├─ checkpoint.mark_completed(...)   # atomic write
                 └─ Rich table printed to console
```

---

## Layer Status

| Layer | File(s) | Status | Notes |
|-------|---------|--------|-------|
| BaseEstimator | `estimators/common/base.py` | ✅ Real | `update()`, `reset()`, `default_config()`, `tuning_spec()`, `structural_latency_samples()` |
| EstimatorSpec / TuningSpec | `estimators/common/types.py` | ✅ Real | Frozen dataclasses, `candidate_grid()`, `generate_grid()` |
| ScenarioBase | `scenarios/_base.py` | ✅ Real | `build()`, `set_montecarlo_tuning()`, `tuning_map` |
| compute_metrics | `metrics/frequency.py` | ✅ Real | 25+ metric keys, IEEE compliance |
| aggregate_monte_carlo | `stats/aggregate.py` | ✅ Real | mean/std/p95/p99/n per metric |
| TimingHarness | `profiling/timing.py` | ✅ Real | External wall-clock, not inside estimator |
| TraceRunner | `runners/trace_runner.py` | ✅ Real | One (scenario × method × seed) |
| ScenarioMethodRunner | `runners/scenario_method_runner.py` | ✅ Real | N-seed MC loop |
| ArtifactStore | `io/artifact_store.py` | ✅ Real | SHA-256 deterministic keys |
| BenchmarkConfig | `core/config_models.py` | ✅ Real | Pydantic v2, YAML loader |
| EstimatorRegistry | `estimators/registry.py` | ✅ Real | 15 registered (9 real, 6 stub) |
| ScenarioRegistry | `scenarios/registry.py` | ✅ Real | 3 registered |
| CheckpointManager | `core/checkpoint.py` | ✅ Real | resume/restart/partial, atomic write |
| `ofb run` | `cli/commands/run.py` | ✅ Real | YAML → MC loop → JSON artifacts |
| `ofb list` | `cli/commands/list_items.py` | ✅ Real | Rich table of registries |
| `ofb doctor` | `cli/commands/doctor.py` | ✅ Real | Import sanity checks |
| `ofb version` | `cli/app.py` | ✅ Real | Version string |
| GridSearch tuning | `tuning/grid_search.py` | ✅ Scaffold | Grid search, not wired to CLI |
| ScenarioRunner | `runners/scenario_runner.py` | ⏳ Stub | Phase 4 |
| SuiteRunner | `runners/suite_runner.py` | ⏳ Stub | Phase 4 |
| WorkerPool | `runners/worker_pool.py` | ⏳ Stub | Phase 4 (parallel) |
| Bootstrap CI | `stats/intervals.py` | ⏳ Stub | Phase 3 |
| Wilcoxon test | `stats/hypothesis.py` | ⏳ Stub | Phase 3 |
| Effect sizes | `stats/effect_sizes.py` | ⏳ Stub | Phase 3 |
| Pareto ranking | `stats/ranking.py` | ⏳ Stub | Phase 3 |
| `ofb analyze` | `cli/commands/analyze.py` | ⏳ Stub | Phase 5 |
| `ofb scaffold` | `cli/commands/scaffold.py` | ⏳ Stub | Phase 1 |
| `ofb status` (full) | `cli/commands/status.py` | ⏳ Stub | Phase 4 |
| Reports | `reports/` | ⏳ Stub | Phase 5 |
| Plotting | `plotting/` | ⏳ Stub | Phase 5 |
| Compat layer | `compat/` | ⏳ Stub | Phase 2 (parity tests) |

---

## Estimator Registry (15 registered)

| Name | Class | Family | Status | Algorithm |
|------|-------|--------|--------|-----------|
| `Baseline_Passthrough` | BaselinePassthrough | control | ✅ Real | Always returns f_nom |
| `SOGI_FLL` | SOGIFLLEstimator | f0_pll | ✅ Real | Second-Order Generalized Integrator FLL |
| `SRF_PLL` | SRFPLLEstimator | f0_pll | ✅ Real | SOGI+SRF-PLL with PI controller, anti-windup |
| `EKF_Freq` | EKFFreqEstimator | f1_kalman | ✅ Real | Extended Kalman Filter (Joseph-form P) |
| `RAEKF` | RAEKFEstimator | f1_kalman | ✅ Real | Robust Adaptive EKF (Huber-M + Sage-Husa R) |
| `FFTPeak` | FFTPeakEstimator | f2_window | ✅ Real | FFT bin peak detection |
| `IpDFT` | IpDFTEstimator | f2_window | ✅ Real | Interpolated DFT (Hann, 2-point) |
| `ZeroCrossing` | ZeroCrossingEstimator | f3_recursive | ✅ Real | Threshold zero-crossing |
| `RDFT` | RDFTEstimator | f3_recursive | ✅ Real | Recursive (sliding) DFT |
| `UKF` | UKFEstimator | f1_kalman | ⏳ Stub | Returns f_nom, valid=False |
| `TFT` | TFTEstimator | f2_window | ⏳ Stub | Returns f_nom, valid=False |
| `RLS` | RLSEstimator | f3_recursive | ⏳ Stub | Returns f_nom, valid=False |
| `RLS_VFF` | RLSVFFEstimator | f3_recursive | ⏳ Stub | Returns f_nom, valid=False |
| `Koopman_RKDPMU` | KoopmanRKDPMUEstimator | f4_data_driven | ⏳ Stub | Returns f_nom, valid=False |
| `PI_GRU` | PIGRUEstimator | f4_data_driven | ⏳ Stub | Returns f_nom, valid=False |

**Still to port (37+ estimators):** See ROADMAP.md Phase 6.

---

## Scenario Registry (3 registered)

| Name | Class | Group | Status | Description |
|------|-------|-------|--------|-------------|
| `G1_E1_Pure_60Hz` | G1_E1_Pure_60Hz | g1 | ✅ Real | 60 Hz sine, configurable noise, duration |
| `G2_E1_FreqStep` | G2_E1_FreqStep | g2 | ✅ Real | Abrupt frequency step at t_step |
| `G2_E2_FreqRamp` | G2_E2_FreqRamp | g2 | ✅ Real | Linear frequency ramp over interval |

**Still to add (15+ scenarios):** See ROADMAP.md Phase 2, 7.

---

## Metric Suite (compute_metrics output keys)

All keys have format `{"value": float, "units": str, "compliance": {...}}`.

| Group | Keys | Status |
|-------|------|--------|
| Accuracy | RMSE_HZ, MAE_HZ, BIAS_HZ, MED_ABS_ERR_HZ, MAD_ABS_ERR_HZ | ✅ |
| IEEE compliance | FE_MAX_MHZ, FE_OUTLIER_RATE, RFE_RMSE_HZS, RFE_MAX_ABS_HZS, RFE_OUTLIER_RATE | ✅ |
| Robust tails | P50_ABS_ERR_HZ, P95_ABS_ERR_HZ, P99_ABS_ERR_HZ, CVAR95_ABS_ERR_HZ | ✅ |
| Dynamics | SETTLING_TIME_S, RESPONSE_TIME_S, OVERSHOOT, UNDERSHOOT, NADIR_HZ | ✅ |
| ROCOF | ROCOF_PEAK_ABS, ROCOF_ERR_MAX_ABS | ✅ |
| Protection | TRIPPED_0p2, TRIP_TIME_0p2_S, TRIPPED_0p5, TRIP_TIME_0p5_S | ✅ |
| Cost | TIME_PER_SAMPLE_US, LATENCY_SAMPLES | ✅ |

---

## Test Suite Status

```
pytest tests/           → 292 passed, 0 failed, 1 warning (scipy precision)

Distribution:
  smoke/         ~20 tests   < 30s    imports, pipeline, CLI
  unit/          ~200 tests  < 2min   per-class isolation
    estimators/  ~160        base interfaces, all 9 real estimators (+RAEKF, SRF_PLL)
    scenarios/   ~20         G1_E1, G2_E2 build/shape/determinism
    metrics/     ~15         compute_metrics known-answer tests
    stats/       ~10         aggregate_monte_carlo
    core/        ~5          run_identity hashing
    checkpoint   ~26         session lifecycle, atomic write
  integration/   ~30 tests   < 5min   cross-layer pipelines
  regression/    ~30 tests   ~10min   parity vs. v1/PMU/pfebench reference
```

---

## Known Issues & Bugs

### P0 — Blocking (must fix before first published comparison)

| # | Issue | File | Impact |
|---|-------|------|--------|
| P0-01 | 8 stub estimators return `valid=False` — excluded from metrics | `estimators/monophasic/*/` | Can't compare stubs to real methods |
| P0-02 | No `G2_E3_PhaseJump` or `G1_E2_NoiseSNR` scenarios | `scenarios/g2/`, `scenarios/g1/` | Missing fundamental test cases |
| P0-03 | No bootstrap CI → no uncertainty quantification | `stats/intervals.py` | Can't make significance claims |
| P0-04 | No statistical hypothesis tests | `stats/hypothesis.py` | Can't claim A > B at p < 0.05 |
| P0-05 | No effect sizes | `stats/effect_sizes.py` | IEEE reviewer requirement |

### P1 — Required for publication

| # | Issue | File | Impact |
|---|-------|------|--------|
| P1-01 | `SuiteRunner` not wired — `ofb run` uses nested loops, not runner hierarchy | `runners/suite_runner.py`, `cli/commands/run.py` | No automated full-suite run |
| P1-02 | `ofb scaffold estimator` not implemented | `cli/commands/scaffold.py` | DX friction when adding new estimators |
| P1-03 | `ofb analyze` not implemented — no post-hoc stats | `cli/commands/analyze.py` | Can't generate paper tables from CLI |
| P1-04 | No plotting at all — no `matplotlib` code runs | `plotting/` | Can't generate paper figures |
| P1-05 | `BENCHMARK_SPEC.md` lacks physics equations | `BENCHMARK_SPEC.md` | Missing scientific specification |
| P1-06 | Pre-registered hypotheses H1–H15 not implemented | `stats/hypothesis_suite.py` | Core scientific claim |

### P2 — Required for JOSS

| # | Issue | File | Impact |
|---|-------|------|--------|
| P2-01 | No parallel execution — 45×18×100 = 81,000 runs, ~22 hours single-core | `runners/worker_pool.py` | Slow |
| P2-02 | No `CITATION.cff` | — | JOSS requirement |
| P2-03 | No GitHub Actions CI | `.github/workflows/` | Deleted in current branch |
| P2-04 | Only 3 of 18 scenarios — insufficient for IEEE Transactions scope | `scenarios/g3/`, `scenarios/g4/` | Limited novelty |
| P2-05 | `compat/` layer not functional — parity tests use reference files directly | `compat/` | Brittle regression tests |

### P3 — Nice-to-have

| # | Issue | File | Impact |
|---|-------|------|--------|
| P3-01 | IpDFT has ~2.3 Hz systematic error (Hann formula vs rectangular formula) | `estimators/monophasic/f2_window/ipdft.py` | Accuracy gap vs optimal IpDFT |
| P3-02 | EKF stability only tested with `clip(P, -1e8, 1e8)` — not guaranteed | `estimators/monophasic/f1_kalman/ekf_freq.py` | Could diverge on edge cases |
| P3-03 | No performance regression tests (CPU µs/sample bounds) | `tests/performance/` | CI doesn't catch performance regressions |
| P3-04 | Docker image not tested | `Dockerfile` | Reproducibility gap |
| P3-05 | No PyPI release | — | Requires manual install |

---

## Artifact Output Schema (what `ofb run` produces today)

```json
{
  "scenario_id": "G1_E1_Pure_60Hz",
  "method_id": "ZeroCrossing",
  "n_runs": 5,
  "seed_start": 0,
  "method_params": {"filter_win": 5},
  "aggregated": {
    "RMSE_HZ": {"mean": 0.0123, "std": 0.0012, "max": 0.018, "p95": 0.017, "p99": 0.018, "n": 5},
    "TIME_PER_SAMPLE_US": {"mean": 0.42, "std": 0.05, ...},
    ...
  },
  "per_seed": [
    {"seed": 0, "RMSE_HZ": 0.012, "MAE_HZ": 0.009, "BIAS_HZ": -0.001, "FE_MAX_MHZ": 12.4, "TIME_PER_SAMPLE_US": 0.41},
    ...
  ]
}
```

---

## Next Actions (ordered by priority)

1. **Port RA-EKF** — the proposed method for the paper (`P0-02`)
2. **Port SRF-PLL** — needed for PLL family comparison (`P0-01`)
3. **Add G1_E2_NoiseSNR and G2_E3_PhaseJump** — needed for M1 (`P0-04, P0-05`)
4. **Implement Bootstrap CI** — needed for any uncertainty claim (`P0-03`)
5. **Implement Wilcoxon test + effect sizes** — needed for significance claims (`P0-04, P0-05`)
6. **Wire SuiteRunner** — needed for automated full run (`P1-01`)
7. **Implement `ofb analyze`** — needed for paper tables (`P1-03`)
8. **Generate paper figures** — needed for submission (`P1-04`)
