# OpenFreqBench — Architecture

> Canonical reference for the `src/openfreqbench/` package design.

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────┐
│  CLI (ofb run / ofb list)                   │
│  src/openfreqbench/cli/                     │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  Runner Layer                               │
│  SuiteRunner → ScenarioRunner              │
│  → ScenarioMethodRunner → TraceRunner       │
└──────┬──────────────────────┬──────────────┘
       │                      │
┌──────▼──────┐    ┌──────────▼──────────────┐
│  Estimators │    │  Scenarios              │
│  _base.py   │    │  _base.py               │
│  zc.py …    │    │  g1/e1_pure_60hz.py …   │
│             │    │                         │
│  numpy only │    │  numpy only             │
└──────┬──────┘    └──────────┬──────────────┘
       │                      │
┌──────▼──────────────────────▼──────────────┐
│  Metrics                                   │
│  frequency.py — compute_metrics()          │
│  aggregate.py — aggregate_monte_carlo()    │
│  numpy only                                │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  I/O + Profiling                            │
│  ArtifactStore, TimingHarness              │
└─────────────────────────────────────────────┘
```

---

## 2. Runner Hierarchy

| Layer | Responsibility |
|-------|---------------|
| `TraceRunner` | One (scenario × method × seed): waveform → f_hat → metrics |
| `ScenarioMethodRunner` | N seeds MC loop → aggregated stats |
| `ScenarioRunner` *(todo)* | All methods for one scenario |
| `SuiteRunner` *(todo)* | All scenarios × all methods |

### TraceRunner internals

```
scenario.set_montecarlo_tuning({"seed": seed})
waveform = scenario.build()
f_hat, exec_time_s = TimingHarness.timed_run(estimator, waveform.v)
metrics = compute_metrics(f_hat, waveform.f_true, exec_time_s, latency, cfg)
→ TraceResult
```

---

## 3. Estimator Contract

```python
class BaseEstimator(ABC):
    NAME: str                     # unique identifier
    FAMILY: str                   # algorithm family

    def _step(self, v: float) -> float:   # PURE MATH
    def reset(self) -> None:              # re-init state
    @property
    def latency_samples(self) -> int:     # causal delay
    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
    def set_params(self, **kwargs) -> None
```

**Invariants:**
- `_step()` must be stateless relative to external calls (pure signal processing)
- No timing inside `_step()` or `step()` — `TimingHarness` wraps `run()`
- No framework imports (pydantic, typer, rich) in estimator files

---

## 4. Scenario Contract

```python
@dataclass
class MyScenario(ScenarioBase):
    scenario_id: str = "G1_E1_..."
    tuning_map: Dict[str, str] = {...}   # alias → attribute

    def build(self) -> ScenarioOutput:
        ...  # generate t, v, f_true, phi, A
```

`set_montecarlo_tuning({"seed": 42, "f": 59.9})` maps domain aliases to internal params safely.

---

## 5. Metric Suite (Q1 grade)

`compute_metrics(f_hat, f_true, exec_time_s, latency, cfg, scenario_id)` produces:

| Group | Metrics |
|-------|---------|
| Steady-state | RMSE_HZ, MAE_HZ, BIAS_HZ, MED_ABS_ERR_HZ, MAD_ABS_ERR_HZ |
| IEEE compliance | FE_MAX_MHZ, FE_OUTLIER_RATE, RFE_RMSE_HZS, RFE_MAX_ABS_HZS, RFE_OUTLIER_RATE |
| Robust tails | P50/P95/P99_ABS_ERR_HZ, CVAR95_ABS_ERR_HZ |
| Event dynamics | SETTLING_TIME, RESPONSE_TIME, OVERSHOOT, UNDERSHOOT, NADIR |
| ROCOF | ROCOF_PEAK_ABS, ROCOF_ERR_MAX_ABS |
| Protection | TRIPPED_0p2, TRIP_TIME_0p2, TRIPPED_0p5, TRIP_TIME_0p5 |
| Cost | TIME_PER_SAMPLE_US, LATENCY_SAMPLES |

All metrics carry `{name, value, units, compliance: {threshold, mode, passed}}`.

---

## 6. Artifact Identity (deterministic cache)

```python
ArtifactIdentity(
    scenario_id, method_id, seed,
    scenario_params_hash,   # sha256[:8] of scenario config dict
    method_params_hash,     # sha256[:8] of method params dict
    schema_version="v1",
)
```

`cache_key` = `v1__<scenario>__<sc_hash>__<method>__<m_hash>__<seed>`

Enables **resume** (skip already-computed seeds) without re-running completed work.

---

## 7. Configuration (YAML-driven)

```yaml
benchmark:
  name: my_run
  output_dir: artifacts
  n_runs: 100
  seed_start: 0

scenarios:
  - id: G1_E1_Pure_60Hz
    params: {fs_hz: 10000, T_s: 5}

estimators:
  - id: ZeroCrossing
    params: {filter_win: 5}

metrics:
  fs_hz: 10000
  f_nom: 60
```

`BenchmarkConfig` (Pydantic v2) validates and parses this at load time.

---

## 8. Registry

```python
EstimatorRegistry.register(ZeroCrossingEstimator)
estimator = EstimatorRegistry.build("ZeroCrossing", params={"filter_win": 10})

ScenarioRegistry.register(G1_E1_Pure_60Hz)
scenario = ScenarioRegistry.build("G1_E1_Pure_60Hz")
```

All built-in registrations happen at import time in `core/registry.py`.
Plugin estimators can call `EstimatorRegistry.register()` from their own package.

---

## 9. Dependency Rules (enforce with linting)

```
# ALLOWED
estimators/ → numpy
scenarios/  → numpy
metrics/    → numpy

# FORBIDDEN in estimators/, scenarios/, metrics/
import pydantic / typer / rich / yaml / pandas / matplotlib

# ALLOWED anywhere in runners/, cli/, io/
all dependencies
```

---

## 10. Ported vs. Pending

| Module | Status | Source |
|--------|--------|--------|
| `ZeroCrossingEstimator` | ✅ Ported | pfebench/e1_zc.py |
| `G1_E1_Pure_60Hz` | ✅ Ported | pfebench/G1_E1_Pure_60Hz.py |
| `compute_metrics` | ✅ Ported | pfebench/metrics/metrics.py |
| `aggregate_monte_carlo` | ✅ Ported | pfebench/metrics/metrics.py |
| All other estimators (45+) | ⏳ Pending | pfebench/estimators/ |
| All other scenarios (18+) | ⏳ Pending | pfebench/scenarios/ |
| TuningRunner (GSO) | ⏳ Pending | extracted from BaseEstimator.optimize() |
| ProcessPoolExecutor | ⏳ Pending | Phase 4 |
| Statistical upgrades | ⏳ Pending | Phase 6 (Bonferroni, Cohen's d) |
