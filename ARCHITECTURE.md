# OpenFreqBench — Architecture Reference
> Canonical technical design for `src/openfreqbench/` | Branch: `q1-arch-final` | Updated: 2026-03-19

---

## 1. Design Philosophy

OpenFreqBench is built around three hard invariants:

1. **Estimators are pure signal-processing plugins.** They receive one voltage sample and return one number. No timing, no logging, no framework imports, no global state.
2. **All measurement is external.** `TimingHarness` wraps `run()` from the outside; the estimator never times itself. `compute_metrics()` is called by the runner, not the estimator.
3. **Reproducibility by construction.** Every artifact is addressed by a deterministic SHA-256 key derived from scenario params, method params, and seed. Re-running with the same inputs produces bit-for-bit identical output.

These invariants allow any estimator to be benchmarked, profiled, tuned, and compared without modifying it, and allow the benchmark to be reproduced years later.

---

## 2. Layer Map

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLI  (typer, rich, yaml)                                            │
│  src/openfreqbench/cli/                                              │
│  ofb run | ofb list | ofb doctor | ofb status | ofb scaffold         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Configuration & Registry                                            │
│  core/config_models.py   — BenchmarkConfig (Pydantic v2, YAML)      │
│  core/registry.py        — EstimatorRegistry, ScenarioRegistry       │
│  core/checkpoint.py      — Resume/restart session management         │
│  core/run_identity.py    — Deterministic SHA-256 artifact keys       │
└──────────┬─────────────────────────────────────────┬─────────────────┘
           │                                         │
┌──────────▼─────────────┐             ┌─────────────▼────────────────┐
│  Runner Layer           │             │  Persistence / I/O           │
│  runners/               │             │  io/artifact_store.py        │
│  ├─ TraceRunner         │             │  io/json_writer.py           │
│  ├─ ScenarioMethodRunner│             │  io/csv_writer.py            │
│  ├─ ScenarioRunner *    │             │  profiling/timing.py         │
│  └─ SuiteRunner *       │             └──────────────────────────────┘
└──────┬───────────┬──────┘
       │           │
┌──────▼──────┐  ┌─▼──────────────────────────────────────────────────┐
│  Estimators │  │  Scenarios                                         │
│  (numpy only│  │  (numpy only)                                      │
│             │  │                                                    │
│  common/    │  │  _base.py — ScenarioBase, ScenarioOutput          │
│  ├─ base.py │  │  common/  — noise, timebase, waveform, envelopes  │
│  └─ types.py│  │  g1/      — steady state & noise                  │
│             │  │  g2/      — dynamic events (step, ramp, jump)     │
│  f0_pll/    │  │  g3/      — IEEE OpenDSS grids *                  │
│  f1_kalman/ │  │  g4/      — real PMU recordings *                 │
│  f2_window/ │  └────────────────────────────────────────────────────┘
│  f3_recursi/│
│  f4_data_d/ │  * = not yet implemented
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
│  stats/aggregate.py    — aggregate_monte_carlo() MC summarisation   │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│  Statistics / Analysis   [Phase 3 — mostly stubs]                   │
│  stats/intervals.py    — Bootstrap CI (5000 resamples)              │
│  stats/hypothesis.py   — Wilcoxon signed-rank, Mann-Whitney U       │
│  stats/effect_sizes.py — Cohen's d, Cliff's delta, rank-biserial r  │
│  stats/ranking.py      — Pareto dominance, Borda count              │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│  Reports & Plotting   [Phase 5 — mostly stubs]                      │
│  reports/markdown.py    — Markdown + LaTeX table generation         │
│  plotting/scenario_plots.py — f_hat vs f_true overlays              │
│  plotting/pareto.py     — RMSE vs CPU scatter, Pareto frontier       │
│  plotting/suite_plots.py — Heatmap (scenarios × estimators)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Single Responsibility Boundaries (Strict)

| Class / Function | Owns | Does NOT own |
|------------------|------|--------------|
| `BaseEstimator._step(v)` | Signal-processing math | Timing, logging, metric computation |
| `TimingHarness.timed_run()` | Wall-clock measurement | Algorithm logic |
| `TraceRunner.run()` | One (scenario × method × seed) execution | MC loop, aggregation, persistence |
| `ScenarioMethodRunner.run()` | N-seed MC loop, aggregation | Per-sample logic, metric definitions |
| `compute_metrics()` | Metric computation from arrays | Estimation, timing, I/O |
| `ArtifactStore.save_json()` | JSON persistence with deterministic keys | Metric computation, result interpretation |
| `CheckpointManager` | Session state (which pairs are done) | Running pairs, metric computation |
| `BenchmarkConfig` | YAML parsing and validation | Execution, metric computation |

---

## 4. Estimator Contract

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

    # Required: declare tunable parameters
    @classmethod
    def default_config(cls) -> dict:
        return {"fs": 10_000.0, "window_size": 1024}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(params=[
            TuningParam(name="window_size", default=1024, type="int",
                        values=[256, 512, 1024, 2048])
        ], objective="RMSE_HZ")

    # Required: re-initialise all state
    def reset(self) -> None:
        self._window_size = int(self._config.get("window_size", 1024))
        self._f_est = self.NOMINAL_FREQ_HZ

    # Required: causal delay in samples
    def structural_latency_samples(self) -> int:
        return self._window_size // 2

    # Required: process one sample
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
| `update(v) → EstimatorOutput` | `step(v) → float` |
| `reconfigure(config)` | `set_params(**kwargs)` |
| `structural_latency_samples()` | `latency_samples` property |
| `tuning_spec().params` | `tuning_ranges()` |
| `_config` dict | `_params` property (alias) |

### Constructor accepts both old and new style

```python
# New style
est = MyEstimator(config={"fs": 10000.0, "window_size": 512})

# Backward-compat (old style, still works)
est = MyEstimator(params={"window_size": 512})
```

---

## 5. Scenario Contract

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

`set_montecarlo_tuning({"seed": 42, "f_end": 59.3})` maps aliases to attributes via `tuning_map` safely.

---

## 6. Runner Hierarchy

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

**Runner status:**

| Runner | Status | Used by |
|--------|--------|---------|
| `TraceRunner` | ✅ Real | `ScenarioMethodRunner`, integration tests |
| `ScenarioMethodRunner` | ✅ Real | `ofb run`, integration tests |
| `ScenarioRunner` | ⏳ Stub | Phase 4 |
| `SuiteRunner` | ⏳ Stub | Phase 4 |
| `WorkerPool` | ⏳ Stub | Phase 4 (parallel) |

---

## 7. Metric Suite

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

## 8. Artifact Identity (Deterministic Cache)

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

## 9. Checkpoint System

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
No partial checkpoint files possible on POSIX/Windows.

CLI integration:

```bash
ofb run config.yaml            # interactive dialog if checkpoint exists
ofb run config.yaml --resume   # auto-resume, no dialog
ofb run config.yaml --restart  # delete checkpoint, start fresh
ofb run config.yaml --status   # show progress and exit
```

---

## 10. Configuration Schema (YAML)

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
  warm_up_s: 0.1            # exclude first warm_up_s from metric computation
  ieee_fe_limit_mhz: 5.0    # IEEE C37.118 frequency error limit (mHz)
  ieee_rfe_limit_hzs: 0.1   # IEEE C37.118 ROCOF error limit (Hz/s)
```

`BenchmarkConfig` (Pydantic v2) validates all fields at load time with clear error messages.

---

## 11. EstimatorSpec and TuningSpec

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

    def generate_grid(self) -> List[Any]:
        """Explicit values, or linspace(low, high, n) for range."""

@dataclass
class TuningSpec:
    params: List[TuningParam]
    objective: str = "RMSE_HZ"
    method: str = "grid"

    def candidate_grid(self) -> List[Dict[str, Any]]:
        """Cartesian product of all parameter grids."""

    def n_candidates(self) -> int:
        """Total number of grid candidates (product of grid sizes)."""
```

---

## 12. Dependency Rules (Enforced)

```
# ALLOWED: bottom layers only use numpy
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

Enforced via `ruff` rules and `mypy --strict` in CI.

---

## 13. Estimator Family Tree

```
src/openfreqbench/estimators/
├── common/
│   ├── base.py           # BaseEstimator (abstract base class)
│   ├── types.py          # EstimatorOutput, EstimatorSpec, TuningParam, TuningSpec
│   └── baseline_passthrough.py   # Always returns f_nom (control baseline)
│
└── monophasic/
    ├── f0_pll/           # Phase-Locked Loops & FLLs
    │   ├── sogi_fll.py   # ✅ SOGI-FLL (Second-Order Generalized Integrator)
    │   └── srf_pll.py    # ⏳ SRF-PLL (Synchronous Reference Frame PLL)
    │
    ├── f1_kalman/        # State-space / Kalman filters
    │   ├── ekf_freq.py   # ✅ EKF (Extended Kalman Filter) — Joseph-form P update
    │   ├── raekf.py      # ⏳ RA-EKF (Robust Adaptive EKF)
    │   └── ukf.py        # ⏳ UKF (Unscented Kalman Filter)
    │
    ├── f2_window/        # Block spectral / windowed methods
    │   ├── fft_peak.py   # ✅ FFT Peak detector
    │   ├── ipdft.py      # ✅ Interpolated DFT (Hann window, 2-point formula)
    │   └── tft.py        # ⏳ Taylor-Fourier Transform
    │
    ├── f3_recursive/     # Sample-by-sample recursive methods
    │   ├── zero_crossing.py  # ✅ Zero-crossing detector
    │   ├── rdft.py           # ✅ Recursive (Sliding) DFT
    │   ├── rls.py            # ⏳ Recursive Least Squares
    │   └── rls_vff.py        # ⏳ RLS with Variable Forgetting Factor
    │
    └── f4_data_driven/   # Machine learning / data-driven
        ├── koopman_rkdpmu.py # ⏳ Koopman-operator EDMD
        └── pi_gru.py         # ⏳ Physics-Informed GRU

Legend: ✅ = real implementation  ⏳ = stub (returns f_nom, valid=False)
```

---

## 14. Scenario Group Map

```
src/openfreqbench/scenarios/
├── g1/     Steady-state & noise          (2 of ~4 planned)
│   └── e1_pure_60hz.py  ✅
│
├── g2/     Dynamic single events          (2 of ~6 planned)
│   ├── e1_freq_step.py  ✅
│   └── e2_freq_ramp.py  ✅
│
├── g3/     IEEE OpenDSS grid scenarios    (0 of 8 planned)
│           [requires pip install -e ".[opendss]"]
│
└── g4/     Real PMU waveform recordings  (0 of 2 planned)
            [requires data files in data/]
```

---

## 15. Three-World Coexistence

The repository maintains three concurrent codebases at different maturity levels:

| Path | Role | Policy |
|------|------|--------|
| `src/openfreqbench/` | Canonical — all new work goes here | Read/write |
| `v1/PMU/pfebench/` | Scientific reference — real Q1 results | Read-only; never modify |
| `legacy/` | Archived skeleton — do not resurrect | Read-only reference |

When porting from `v1/PMU/pfebench/`:
1. Port algorithm logic to new `BaseEstimator` API
2. Write parity regression test vs. `v1/PMU/artifacts/` reference values
3. Do NOT modify the reference files
