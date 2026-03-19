# OpenFreqBench — Architecture Reference

> Canonical technical design for `src/openfreqbench/` | Branch: `q1-arch-final` | Updated: 2026-03-19

---

## 1. Design Philosophy

OpenFreqBench is built around three hard invariants:

1. **Estimators are pure signal-processing plugins.** They receive one voltage sample and return one typed result. No timing, no logging, no framework imports, no global state.
2. **All measurement is external.** `TimingHarness` wraps `run()` from the outside; the estimator never times itself. `compute_metrics()` is called by the runner, not the estimator.
3. **Reproducibility by construction.** Every artifact is addressed by a deterministic SHA-256 key derived from scenario params, method params, and seed. Re-running with the same inputs produces bit-for-bit identical output.

These invariants allow any estimator to be benchmarked, profiled, tuned, and compared without modifying it.

---

## 2. Repository Layout

```
open-freq-bench/
├── src/openfreqbench/      ← CANONICAL PACKAGE — all work goes here
├── v1/PMU/pfebench/        ← SCIENTIFIC REFERENCE — read-only
├── v1/PMU/artifacts/       ← REAL Q1 BENCHMARK RESULTS — never delete
├── legacy/                 ← Archived skeletons — .gitkeep only
├── tests/                  ← Test suite (43 test files)
├── examples/               ← YAML config examples and notebooks
├── scripts/                ← Shell-based dev / CI scripts
├── docs/                   ← Documentation sources
├── data/                   ← Reference data files
└── [config files]          ← pyproject.toml, mypy.ini, .ruff.toml, etc.

REMOVED (2026-03-19):
  estimators/               ← Was legacy dead code (broken utils.pmu imports)
  scenarios/                ← Was legacy empty stubs (superseded by src/)
  evaluation/               ← Was dead code (superseded by src/metrics/)
  core/pmu/                 ← Was 3 empty placeholder files
  pipelines/                ← Was broken runner (superseded by src/runners/)
  ofb/                      ← Was stub registry (superseded by src/core/registry.py)
```

---

## 3. Layer Map

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLI  (typer, rich, yaml)                                            │
│  src/openfreqbench/cli/                                              │
│  ofb run | ofb list | ofb doctor | ofb status | ofb scaffold         │
│  ofb analyze | ofb smoke | ofb version                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Configuration & Registry                                            │
│  core/config_models.py   — BenchmarkConfig (Pydantic v2, YAML)      │
│  core/registry.py        — EstimatorRegistry, ScenarioRegistry       │
│  core/checkpoint.py      — Resume/restart session management         │
│  core/run_identity.py    — Deterministic SHA-256 artifact keys       │
│  core/constants.py       — NOMINAL_FREQ_HZ, limits                  │
│  core/enums.py           — RunMode, EstimatorStatus                  │
│  core/errors.py          — BenchmarkError hierarchy                  │
│  core/hashing.py         — Deterministic SHA-256 helpers             │
│  core/ids.py             — ID normalization                          │
│  core/metadata.py        — Run metadata containers                   │
│  core/paths.py           — Artifact path conventions                 │
│  core/types.py           — Shared type aliases                       │
└──────────┬─────────────────────────────────────────┬─────────────────┘
           │                                         │
┌──────────▼─────────────┐             ┌─────────────▼────────────────┐
│  Runner Layer           │             │  Persistence / I/O           │
│  runners/               │             │  io/artifact_store.py        │
│  ├─ TraceRunner    ✅   │             │  io/json_writer.py           │
│  ├─ ScenarioMethodRunner✅            │  io/csv_writer.py            │
│  ├─ SmokeBenchmarkRunner✅            │  io/layout.py                │
│  ├─ ScenarioRunner  ⏳  │             │  io/readers.py               │
│  ├─ SuiteRunner     ⏳  │             │  io/serializers.py           │
│  └─ WorkerPool      ⏳  │             │  io/snapshots.py             │
└──────┬───────────┬──────┘             │  profiling/timing.py         │
       │           │                    │  profiling/memory.py         │
┌──────▼──────┐  ┌─▼──────────────────────────────────────────────────┐
│  Estimators │  │  Scenarios                                         │
│  (numpy only│  │  (numpy only)                                      │
│             │  │                                                    │
│  common/    │  │  _base.py  — ScenarioBase, ScenarioOutput         │
│  ├─ base.py │  │  common/   — noise, timebase, waveform, envelopes │
│  └─ types.py│  │  g1/ — steady state & noise (3 real, 0 stubs)    │
│             │  │  g2/ — dynamic events (3 real, 6 stubs)           │
│  f0_pll/    │  │  g3/ — interference & modulation (0 real, 6 stubs)│
│  f1_kalman/ │  │  g4/ — composite events (0 real, 3 stubs)         │
│  f2_window/ │  │  g5/ — parameter sweeps (0 real, 2 stubs)         │
│  f3_recursi/│  └────────────────────────────────────────────────────┘
│  f4_data_d/ │
│  f5_param/  │
│  f6_tf/     │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│  Metrics  (numpy only)                                              │
│  metrics/frequency.py  — compute_metrics() master function          │
│  metrics/dynamics.py   — settling time, response time, overshoot   │
│  metrics/rocof.py      — rate-of-change-of-frequency               │
│  metrics/protection.py — under-frequency relay trip thresholds      │
│  metrics/compliance.py — IEEE C37.118.1-2011 standard checks       │
│  metrics/cost.py       — CPU time, memory, structural latency       │
│  metrics/bundles.py    — Metric group bundles                       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│  Statistics / Analysis                                              │
│  stats/aggregate.py    — aggregate_monte_carlo() MC summarisation   │
│  stats/intervals.py    — Bootstrap CI (5000 resamples)              │
│  stats/hypothesis.py   — Wilcoxon signed-rank, Mann-Whitney U       │
│  stats/effect_sizes.py — Cohen's d, Cliff's delta, rank-biserial r  │
│  stats/ranking.py      — Pareto dominance, Borda count              │
│  stats/models.py       — Statistical result containers              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│  Reports & Plotting                                                 │
│  reports/markdown.py        — Markdown + LaTeX table generation     │
│  reports/tables.py          — Rich table builders                   │
│  reports/comparison.py      — Cross-estimator comparison reports    │
│  reports/scenario_report.py — Per-scenario report                   │
│  reports/suite_report.py    — Full benchmark suite report           │
│  plotting/scenario_plots.py — f_hat vs f_true overlays              │
│  plotting/mc_summary.py     — Monte Carlo distribution plots        │
│  plotting/pareto.py         — RMSE vs CPU scatter, Pareto frontier  │
│  plotting/suite_plots.py    — Scenario-overview + benchmark grid    │
│  plotting/styles.py         — IEEE-style matplotlib theme           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Single Responsibility Boundaries (Strict)

| Class / Function | Owns | Does NOT own |
|------------------|------|--------------|
| `BaseEstimator.update(v, ts)` | Signal-processing math | Timing, logging, metric computation |
| `TimingHarness.timed_run()` | Wall-clock measurement | Algorithm logic |
| `TraceRunner.run()` | One (scenario × method × seed) execution | MC loop, aggregation, persistence |
| `ScenarioMethodRunner.run()` | N-seed MC loop, aggregation | Per-sample logic, metric definitions |
| `compute_metrics()` | Metric computation from arrays | Estimation, timing, I/O |
| `ArtifactStore.save_json()` | JSON persistence with deterministic keys | Metric computation, result interpretation |
| `CheckpointManager` | Session state (which pairs are done) | Running pairs, metric computation |
| `BenchmarkConfig` | YAML parsing and validation | Execution, metric computation |
| `EstimatorRegistry` | Estimator class → ID mapping | Instantiation logic, tuning |
| `ScenarioRegistry` | Scenario class → ID mapping | Waveform generation, MC tuning |

---

## 5. Estimator Contract

### Class-level declaration

```python
class MyEstimator(BaseEstimator):
    # Required: set via EstimatorSpec
    SPEC = EstimatorSpec(
        name="MyEst",
        family="spectral",
        family_path="monophasic/f2_window",
        complexity="O(N log N)",
        latency_type="semi-causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> dict:
        return {"fs": 10_000.0, "window_size": 1024}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(params=[
            TuningParam(name="window_size", default=1024, type="int",
                        values=[256, 512, 1024, 2048])
        ], objective="RMSE_HZ")

    def reset(self) -> None:
        self._window_size = int(self._config.get("window_size", 1024))
        self._f_est = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return self._window_size // 2

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        ...
        return EstimatorOutput(frequency_hz=f_est, valid=True)
```

### Auto-derived class attributes (from `__init_subclass__`)

When a subclass sets `SPEC`, the following class attributes are auto-derived:

```
NAME, FAMILY, FAMILY_PATH, COMPLEXITY, LATENCY_TYPE,
NOMINAL_FREQ_HZ, MIN_VALID_FREQ_HZ, MAX_VALID_FREQ_HZ
```

### Backward-compatibility shims (do not use in new code)

| New API | Old API shim |
|---------|-------------|
| `update(v, ts) → EstimatorOutput` | `step(v) → float` |
| `reconfigure(config)` | `set_params(**kwargs)` |
| `structural_latency_samples()` | `latency_samples` property |
| `tuning_spec().params` | `tuning_ranges()` |
| `_config` dict | `_params` property (alias) |

---

## 6. Scenario Contract

```python
@dataclass
class G2_E1_FreqStep(ScenarioBase):
    scenario_id: str = "G2_E1_FreqStep"
    fs_hz: float = 10_000.0
    T_s: float = 2.0
    f_start: float = 60.0
    f_end: float = 59.5
    t_step: float = 1.0
    seed: int = 42

    tuning_map: ClassVar[Dict[str, str]] = {
        "seed": "seed",
        "f_end": "f_end",
    }

    def build(self) -> ScenarioOutput:
        rng = np.random.default_rng(self.seed)
        t   = np.linspace(0, self.T_s, int(self.fs_hz * self.T_s), endpoint=False)
        f_true = np.where(t < self.t_step, self.f_start, self.f_end)
        phi = np.cumsum(2 * np.pi * f_true / self.fs_hz)
        v   = np.sin(phi) + rng.normal(0, 0.01, len(t))
        return ScenarioOutput(
            v=v, t=t, f_true=f_true, phi=phi, A=np.ones_like(t),
            state=ScenarioState(fs_hz=self.fs_hz, f_nom_hz=60.0, T_s=self.T_s),
            scenario_id=self.scenario_id,
        )
```

`set_montecarlo_tuning({"seed": 42, "f_end": 59.3})` maps aliases to attributes via `tuning_map`.

---

## 7. Runner Hierarchy

```
SuiteRunner.run(config)                    # all scenarios × all methods
  └─ ScenarioRunner.run(scenario, config)  # all methods for one scenario
       └─ ScenarioMethodRunner.run(...)    # N seeds MC loop
            └─ TraceRunner.run(seed)       # one seed

TraceRunner.run(scenario, estimator, seed):
  1. scenario.set_montecarlo_tuning({"seed": seed})
  2. waveform = scenario.build()
  3. f_hat, exec_time_s = TimingHarness.timed_run(estimator, waveform.v)
  4. latency = estimator.latency_samples
  5. metrics = compute_metrics(f_hat, f_true, exec_time_s, latency, cfg)
  6. → TraceResult

ScenarioMethodRunner.run(scenario, estimator, n_runs):
  1. [TraceRunner.run(seed) for seed in seeds]
  2. aggregate_monte_carlo(traces)   # → mean/std/p95/p99/n per metric
  3. → ScenarioMethodResult
```

**Runner implementation status:**

| Runner | Status | Used by |
|--------|--------|---------|
| `TraceRunner` | ✅ Real | `ScenarioMethodRunner`, integration tests |
| `ScenarioMethodRunner` | ✅ Real | `ofb run`, integration tests |
| `SmokeBenchmarkRunner` | ✅ Real | `ofb smoke` |
| `ScenarioRunner` | ⏳ Stub | Phase 4 |
| `SuiteRunner` | ⏳ Stub | Phase 4 |
| `WorkerPool` | ⏳ Stub | Phase 4 (parallel) |

---

## 8. Metric Suite

`compute_metrics(f_hat, f_true, exec_time_s, latency_samples, cfg, scenario_id)` → dict

Each metric is a dict: `{"value": float, "units": str, "compliance": {...}}`

### Metric groups

| Group | Keys | IEEE Ref |
|-------|------|---------|
| **Accuracy** | `RMSE_HZ`, `MAE_HZ`, `BIAS_HZ`, `MED_ABS_ERR_HZ`, `MAD_ABS_ERR_HZ` | C37.118.1 §6.2 |
| **Compliance** | `FE_MAX_MHZ`, `FE_OUTLIER_RATE`, `RFE_RMSE_HZS`, `RFE_MAX_ABS_HZS`, `RFE_OUTLIER_RATE` | C37.118.1 §6.3 |
| **Robust tails** | `P50_ABS_ERR_HZ`, `P95_ABS_ERR_HZ`, `P99_ABS_ERR_HZ`, `CVAR95_ABS_ERR_HZ` | — |
| **Dynamics** | `SETTLING_TIME_S`, `RESPONSE_TIME_S`, `OVERSHOOT`, `UNDERSHOOT`, `NADIR_HZ` | C37.118.1 §5 |
| **ROCOF** | `ROCOF_PEAK_ABS`, `ROCOF_ERR_MAX_ABS` | C37.118.1 §7 |
| **Protection** | `TRIPPED_0p2`, `TRIP_TIME_0p2_S`, `TRIPPED_0p5`, `TRIP_TIME_0p5_S` | NERC PRC-024 |
| **Cost** | `TIME_PER_SAMPLE_US`, `LATENCY_SAMPLES` | — |

---

## 9. Artifact Identity (Deterministic Cache)

Every result file has a globally unique, deterministic address:

```python
ArtifactIdentity(
    scenario_id   = "G1_E1_Pure_60Hz",
    method_id     = "ZeroCrossing",
    seed          = 42,
    scenario_params_hash = sha256(scenario_params)[:8],   # e.g. "a3f1c8d2"
    method_params_hash   = sha256(method_params)[:8],     # e.g. "b9e4d071"
    schema_version       = "v1",
)

cache_key = "v1__G1_E1_Pure_60Hz__a3f1c8d2__ZeroCrossing__b9e4d071__42"
file_path = "artifacts/G1_E1_Pure_60Hz/ZeroCrossing/ZeroCrossing_20260319_120000_report.json"
```

This enables:
- **Resume**: `ArtifactStore.exists(identity)` before re-running a seed
- **Deduplication**: same params → same key → cache hit
- **Reproducibility**: key encodes all inputs

---

## 10. Checkpoint System

```
CheckpointManager(output_dir)
  ├─ .ofb_checkpoint.json     # session state (JSON)
  │   ├─ session_id, started_at, config_hash
  │   ├─ completed: {method::scenario: {n_mc, result_file, completed_at}}
  │   ├─ in_progress: {pair, started_at}
  │   └─ failed: {pair: {error, failed_at}}
  │
  ├─ startup_dialog(config, n_total, auto_resume, auto_restart)
  │   → "resume" | "restart" | "partial"
  │
  ├─ mark_started(method, scenario)
  ├─ mark_completed(method, scenario, result_file, n_mc)   ← atomic write
  ├─ mark_failed(method, scenario, error_msg)
  └─ should_run(method, scenario, mode) → bool
```

Atomic write protocol: write to `.ofb_checkpoint.tmp` → `os.replace()`.

CLI integration:

```bash
ofb run config.yaml            # interactive dialog if checkpoint exists
ofb run config.yaml --resume   # auto-resume, no dialog
ofb run config.yaml --restart  # delete checkpoint, start fresh
ofb run config.yaml --status   # show progress and exit
```

---

## 11. Configuration Schema (YAML)

```yaml
benchmark:
  name: "g1_baseline"
  output_dir: artifacts/
  n_runs: 100             # Monte Carlo seeds per (scenario × estimator)
  seed_start: 0           # first seed (seeds = range(seed_start, seed_start + n_runs))
  plots: false            # generate time-series diagnostic plots

scenarios:
  - id: G1_E1_Pure_60Hz
    params:
      fs_hz: 10000
      T_s: 5.0

estimators:
  - id: ZeroCrossing
    params:
      filter_win: 5
  - id: EKF_Freq
    params: {}

metrics:
  fs_hz: 10000
  f_nom: 60.0
  warm_up_s: 0.1
  ieee_fe_limit_mhz: 5.0
  ieee_rfe_limit_hzs: 0.1
```

`BenchmarkConfig` (Pydantic v2) validates all fields at load time with clear error messages.

---

## 12. EstimatorSpec and TuningSpec

```python
@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    family: str
    family_path: str
    complexity: str         # e.g. "O(1)", "O(N log N)"
    latency_type: str       # "causal" | "semi-causal" | "non-causal"
    nominal_freq_hz: float = 60.0
    min_valid_freq_hz: float = 40.0
    max_valid_freq_hz: float = 80.0

@dataclass
class TuningParam:
    name: str
    default: Any
    type: Literal["float", "int", "bool", "str"]
    values: Optional[List[Any]] = None     # explicit search grid
    range: Optional[Tuple[float,float,int]] = None  # (low, high, n_points)
    description: str = ""

@dataclass
class TuningSpec:
    params: List[TuningParam]
    objective: str = "RMSE_HZ"
    method: str = "grid"

    def candidate_grid(self) -> List[Dict[str, Any]]:
        """Cartesian product of all parameter grids."""

    def n_candidates(self) -> int:
        """Total number of grid candidates."""
```

---

## 13. Dependency Rules (Enforced)

```
# ALLOWED: bottom layers — numpy only
estimators/     → numpy, math, collections, abc
scenarios/      → numpy, math, dataclasses
metrics/        → numpy, math

# ALLOWED: middle layers add scipy, pandas
stats/          → numpy, scipy.stats, pandas
tuning/         → numpy, scipy.optimize
profiling/      → time, tracemalloc, psutil

# ALLOWED: top layers add everything
runners/        → numpy, pandas, pydantic, openfreqbench.*
io/             → json, csv, pathlib, openfreqbench.core.*
cli/            → typer, rich, yaml, openfreqbench.*
reports/        → jinja2, markdown
plotting/       → matplotlib

# FORBIDDEN in estimators/, scenarios/, metrics/:
import pydantic
import typer
import rich
import yaml
import pandas
import matplotlib
import torch      # (except f4_data_driven/ with optional dep)
```

Enforced by `ruff` rules and `mypy --strict` target in CI.

---

## 14. Complete Estimator Family Tree

```
src/openfreqbench/estimators/
├── common/
│   ├── base.py                  # BaseEstimator (abstract, canonical)
│   ├── types.py                 # EstimatorOutput, EstimatorSpec, TuningParam, TuningSpec
│   ├── ml_base.py               # BaseMLEstimator (extends BaseEstimator)
│   ├── ml_types.py              # ML-specific types and specs
│   └── baseline_passthrough.py  # Always returns f_nom (control baseline) ✅
│
├── families/                    # Family descriptor modules
│   ├── model_based.py
│   ├── spectral.py
│   └── time_domain.py
│
└── monophasic/
    ├── f0_pll/           Phase-Locked Loops & FLLs
    │   ├── sogi_fll.py   ✅ SOGI-FLL (Second-Order Generalized Integrator)
    │   ├── srf_pll.py    ✅ SRF-PLL (Synchronous Reference Frame PLL)
    │   ├── anf.py        ⏳ ANF (Adaptive Notch Filter)
    │   ├── ddsrf_pll.py  ⏳ DDSRF-PLL (Decoupled Double SRF)
    │   ├── dsogi_fll.py  ⏳ DSOGI-FLL (Dual SOGI)
    │   └── epll.py       ⏳ EPLL (Enhanced PLL)
    │
    ├── f1_kalman/        State-space / Kalman filters
    │   ├── ekf_freq.py   ✅ EKF (Extended Kalman Filter — Joseph-form P update)
    │   ├── raekf.py      ✅ RA-EKF (Robust Adaptive EKF with Huber M-weighting)
    │   ├── ukf.py        ✅ UKF (Unscented Kalman Filter)
    │   ├── adaptive_ekf.py  ⏳ Adaptive EKF (autotuned Q/R)
    │   ├── ckf.py        ⏳ CKF (Cubature Kalman Filter)
    │   └── linear_kalman.py ⏳ Linear KF (linearized state model)
    │
    ├── f2_window/        Block spectral / windowed methods
    │   ├── fft_peak.py   ✅ FFT Peak (parabolic interpolation)
    │   ├── ipdft.py      ✅ Interpolated DFT (Hann window, 2-point formula)
    │   ├── zero_crossing.py ✅ Zero-crossing (threshold + linear interpolation)
    │   ├── dynamic_phasor.py ⏳ Dynamic Phasor (Taylor-Fourier)
    │   ├── goertzel.py   ⏳ Goertzel (single-frequency DFT)
    │   └── tft.py        ✅ TFT (Taylor-Fourier Transform)
    │
    ├── f3_recursive/     Sample-by-sample recursive methods
    │   ├── zero_crossing.py ✅ Zero-Crossing (recursive variant)
    │   ├── rdft.py       ✅ Recursive (Sliding) DFT
    │   ├── rls.py        ✅ RLS (Recursive Least Squares)
    │   ├── rls_vff.py    ✅ RLS-VFF (Variable Forgetting Factor)
    │   ├── interp_zero_crossing.py ⏳ Interpolated ZC
    │   └── windowed_ls.py ⏳ Windowed Least Squares
    │
    ├── f4_data_driven/   Machine learning / data-driven
    │   ├── koopman_rkdpmu.py  ✅ Koopman-operator EDMD
    │   ├── pi_gru.py          ✅ Physics-Informed GRU
    │   ├── bilstm_regressor.py ⏳ BiLSTM Regressor
    │   ├── gru_regressor.py   ⏳ GRU Regressor
    │   ├── lstm_regressor.py  ⏳ LSTM Regressor
    │   ├── mlp_window.py      ⏳ MLP Window
    │   ├── reservoir_echo.py  ⏳ Echo State Reservoir
    │   ├── tcn_frequency.py   ⏳ TCN Frequency
    │   ├── temporal_cnn.py    ⏳ Temporal CNN
    │   └── transformer_regressor.py ⏳ Transformer Regressor
    │
    ├── f5_parametric/    Subspace / parametric methods
    │   ├── esprit.py     ⏳ ESPRIT
    │   ├── music.py      ⏳ MUSIC
    │   └── prony.py      ⏳ Prony method
    │
    ├── f6_time_frequency/ Time-frequency methods
    │   └── hilbert_freq.py ⏳ Hilbert transform instantaneous frequency
    │
    └── f7_hybrid/        Hybrid combinations (reserved, empty)

Shims (backward compat, do not use in new code):
  estimators/_base.py        → re-exports from common/base.py
  estimators/baseline_passthrough.py → re-exports from common/
  estimators/zero_crossing.py → re-exports ZeroCrossingEstimator
  estimators/registry.py     → EstimatorRegistry + all registrations

Legend: ✅ = real implementation  ⏳ = scaffold (returns f_nom, raises NotImplementedError)
```

**Implementation count:** 13 real / 23 scaffolds / 1 baseline = 37 estimators total

---

## 15. Complete Scenario Group Map

```
src/openfreqbench/scenarios/
├── _base.py         — ScenarioBase, ScenarioState, ScenarioOutput
├── _aliases.py      — Scenario ID aliases
├── _outputs.py      — Output type helpers
├── common/
│   ├── envelopes.py — Amplitude/frequency envelope functions
│   ├── noise.py     — White Gaussian, colored, harmonic noise
│   ├── timebase.py  — Sampling grid construction
│   ├── utils.py     — Phase integration, signal helpers
│   └── waveform.py  — Generic waveform builder
│
├── g1/  Steady-state & noise (3 scenarios)
│   ├── e1_pure_60hz.py          ✅ Pure 60 Hz sinusoid, no noise
│   ├── e2_gaussian_noise_1pct.py ⏳ 1% Gaussian noise
│   └── e3_gaussian_noise_5pct.py ⏳ 5% Gaussian noise
│
├── g2/  Dynamic single events (9 scenarios)
│   ├── e1_freq_step.py          ✅ Frequency step +/-0.5 Hz
│   ├── e2_freq_ramp.py          ✅ Frequency ramp at configurable rate
│   ├── e4_voltage_mag_step_1pct.py  ⏳ 1% voltage magnitude step
│   ├── e5_voltage_mag_step_10pct.py ⏳ 10% voltage magnitude step
│   ├── e6_freq_step_59p5.py     ⏳ Freq step to 59.5 Hz
│   ├── e7_freq_step_55.py       ⏳ Freq step to 55 Hz (under-freq)
│   ├── e8_fast_ramp.py          ⏳ Fast ramp (>1 Hz/s)
│   └── e9_slow_ramp.py          ⏳ Slow ramp (<0.1 Hz/s)
│
├── g3/  Interference & modulation (6 scenarios)
│   ├── e10_am_modulation.py     ⏳ Amplitude modulation
│   ├── e11_fm_modulation.py     ⏳ Frequency modulation (flicker)
│   ├── e12_phase_jump.py        ⏳ Phase angle jump
│   ├── e13_impulsive_outliers.py ⏳ Impulsive noise / spike contamination
│   ├── e14_noise_harmonics.py   ⏳ Harmonic distortion + noise
│   └── e15_noise_interharmonics.py ⏳ Interharmonic contamination
│
├── g4/  Composite / real-world events (3 scenarios)
│   ├── e16_composite_islanding.py ⏳ Islanding detection scenario
│   ├── e17_multi_event_profile.py ⏳ Multi-event sequence
│   └── e18_chamorro_event.py    ⏳ Real grid disturbance replay
│
└── g5/  Parameter sweeps (2 scenarios)
    ├── e19_phase_jump_sweep.py  ⏳ Phase jump magnitude sweep
    └── e20_snr_sweep.py         ⏳ SNR sweep

Legend: ✅ = real build()  ⏳ = scaffold (raises NotImplementedError)
```

**Implementation count:** 3 real / 19 scaffolds = 22 scenarios total

---

## 16. Three-World Coexistence Policy

The repository maintains three concurrent codebases at different maturity levels:

| Path | Role | Policy |
|------|------|--------|
| `src/openfreqbench/` | Canonical — all new work goes here | Read/write |
| `v1/PMU/pfebench/` | Scientific reference — real Q1 results | **Read-only; never modify** |
| `v1/PMU/artifacts/` | Benchmark artifacts — measured results | **Never delete or regenerate** |
| `legacy/` | Archived skeletons | Read-only; do not resurrect |

**When porting from `v1/PMU/pfebench/`:**
1. Port algorithm logic to new `BaseEstimator` API
2. Write parity regression test vs. `v1/PMU/artifacts/` reference values
3. Do NOT modify the reference files

**Dead code removed (2026-03-19):**
These root-level directories no longer exist. All functionality is in `src/openfreqbench/`:

| Removed | Why | Replacement |
|---------|-----|-------------|
| `estimators/` | Broken `utils.pmu` imports; unusable | `src/openfreqbench/estimators/` |
| `scenarios/` | Empty placeholder stubs | `src/openfreqbench/scenarios/` |
| `evaluation/` | Dead code, superseded | `src/openfreqbench/metrics/` |
| `core/pmu/` | 3 empty files | N/A |
| `pipelines/` | Broken runner stub | `src/openfreqbench/runners/` |
| `ofb/` | Superseded registry stub | `src/openfreqbench/core/registry.py` |

---

## 17. Compat Layer

`src/openfreqbench/compat/` provides read-side bridges to `v1/PMU/pfebench/` artifacts:

```
compat/adapters.py          — Load pfebench results into canonical types
compat/pfebench_ids.py      — ID mapping: pfebench names → canonical IDs
compat/pfebench_metrics.py  — pfebench metric names → canonical metric keys
compat/pfebench_scenarios.py — pfebench scenario names → canonical scenario IDs
```

This layer is complete but not yet wired into the CLI. Intended use: load `v1/PMU/artifacts/`
into the canonical stats/reporting pipeline for cross-validation.

---

## 18. Known Open Issues (from audit 2026-03-19)

| Issue | Status | Detail |
|-------|--------|--------|
| `mypy src` | PARTIAL | 89 errors remain in `plotting/`, `cli/commands/smoke.py`, `__main__.py` |
| `CODE_OF_CONDUCT.md` | MISSING | Required for JOSS open-source release |
| `CITATION.cff` | MISSING | Required for academic citation |
| `.editorconfig` | MISSING | Required for cross-platform line ending enforcement |
| `.gitattributes` | PARTIAL | Only covers `*.sh`; needs `* text=auto eol=lf` |
| Logger utility | OPEN | `src/openfreqbench/utils/log.py` not implemented |
| `tmp_artifacts_dir` fixture | MISSING | Not in `tests/conftest.py` |
| Scenario stubs (g2-g5) | 19 stubs | Need real `build()` implementations |
| Estimator stubs (f0-f6) | 23 stubs | Need real `update()` implementations |
| `ScenarioRunner` | Stub | Phase 4 |
| `SuiteRunner` | Stub | Phase 4 |
| `WorkerPool` | Stub | Phase 4 (parallel execution) |

---

## 19. File Count Summary

```
src/openfreqbench/
  Core modules              12  (core/ + __main__.py + _version.py)
  CLI modules               12  (app, 8 commands, common, main, render)
  Compat modules             4
  Estimators               132  (common + families + monophasic + shims + registry)
  I/O modules                7
  Metrics modules            7
  Plotting modules           5
  Profiling modules          4
  Reports modules            5
  Runners                    7
  Scenarios                 30  (base + common + g1-g5 + registry)
  Stats modules              6
  Tuning modules             4
  ─────────────────────────────
  TOTAL:                   177 Python files

tests/
  Smoke                      3
  Unit                      27  (after removing 2 broken tests)
  Integration                4
  Regression                 2
  Fixtures                   3
  conftest.py                1
  checkpoint test            1
  ─────────────────────────────
  TOTAL:                    41 test files  (631 passing, 69 pre-existing stub failures)
```
