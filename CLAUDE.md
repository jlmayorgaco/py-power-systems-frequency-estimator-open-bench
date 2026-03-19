# CLAUDE.md — OpenFreqBench Developer Guide for AI Assistants

This file gives Claude Code the context it needs to work effectively in this repository.
Read this before making any edits.

---

## What this project is

**OpenFreqBench** is a Q1-grade benchmark framework for dynamic frequency estimation in power systems.
It compares 45+ algorithms across 18+ scenarios using statistically rigorous metrics (ANOVA, bootstrap CI, CVaR-95).

Target publication: IEEE Transactions / JOSS open-source release.

---

## Critical architecture facts

### Three-world problem (know this first)

The repo has three coexisting code bases:

| Path | Status | Role |
|------|--------|------|
| `src/openfreqbench/` | ✅ **Canonical — write here** | New package, real implementations |
| `v1/PMU/pfebench/` | 🔬 **Scientific reference** | Real science, do NOT delete |
| `ofb/` | ⚠️ Legacy skeleton | Stub runner (`TVE_mean: 0.0`), keep for now |

### Never touch these without explicit instruction
- `v1/PMU/pfebench/` — scientific source of truth; treat as read-only reference
- `v1/PMU/artifacts/` — real Q1 benchmark results; do not regenerate or delete

### The critical bug (P0-01)
`ofb/benchmarks/runner.py` returns hardcoded `TVE_mean: 0.0`. It is a stub, not science.
All real work goes into `src/openfreqbench/`.

---

## Package layout (canonical)

```
src/openfreqbench/
├── _version.py            # single source of version truth
├── __init__.py
├── estimators/
│   ├── _base.py           # BaseEstimator (NO timing, NO GSO)
│   └── zc.py              # ZeroCrossingEstimator (first port)
├── scenarios/
│   ├── _base.py           # ScenarioBase, ScenarioOutput, ScenarioState
│   └── g1/
│       └── e1_pure_60hz.py  # G1_E1_Pure_60Hz (first port)
├── metrics/
│   ├── frequency.py       # compute_metrics(), MetricConfig, math core
│   └── aggregate.py       # aggregate_monte_carlo()
├── profiling/
│   └── timing.py          # TimingHarness (external timing)
├── io/
│   └── artifact_store.py  # ArtifactIdentity, ArtifactStore
├── runners/
│   ├── trace_runner.py    # TraceRunner (one seed)
│   └── scenario_method_runner.py  # MC loop
├── core/
│   ├── config.py          # Pydantic BenchmarkConfig + YAML loader
│   └── registry.py        # EstimatorRegistry, ScenarioRegistry
└── cli/
    ├── app.py             # Typer app → `ofb` command
    └── commands/
        ├── run.py         # `ofb run <config.yaml>`
        └── list_cmd.py    # `ofb list`
```

---

## Design rules (enforce strictly)

### Dependency directions
```
cli → runners → estimators/scenarios/metrics
estimators/ → numpy only (no framework imports)
scenarios/  → numpy only (no framework imports)
metrics/    → numpy only (no framework imports)
```

### SRP boundaries
- `estimator._step(v)` = pure math, no timing, no logging
- `TimingHarness` = measures wall-clock externally
- `TuningRunner` (not yet created) = GSO externally, not in BaseEstimator
- `TraceRunner` = one (scenario × method × seed)
- `ScenarioMethodRunner` = N seeds (MC loop)
- `ArtifactStore` = persistence only

### BaseEstimator contract
```python
_step(v_sample: float) -> float     # pure math
reset() -> None                     # re-init state
latency_samples -> int              # property
tuning_ranges() -> List[TuningParam]  # classmethod
set_params(**kwargs) -> None        # update + reset
```

Timing is measured by `TimingHarness.timed_run()`, NOT inside `step()`.

### ScenarioBase contract
```python
build() -> ScenarioOutput    # generates waveform
set_montecarlo_tuning(params) -> ScenarioBase  # MC param injection
tuning_map: Dict[str, str]  # alias → attribute
```

---

## CLI

```bash
pip install -e ".[dev]"
ofb run examples/quick_smoke.yaml
ofb list
ofb version
```

---

## Tests

```bash
pytest tests/smoke/          # smoke: < 30 s, no tuning
pytest tests/unit/           # unit: individual classes
pytest                       # all tests
```

Test philosophy:
- Smoke tests: real science pipeline, short waveforms, few seeds
- Unit tests: one class at a time, deterministic, fast
- NO mocking of numpy or estimator internals

---

## Adding a new estimator

1. Create `src/openfreqbench/estimators/<name>.py`
2. Subclass `BaseEstimator`, implement `_step`, `reset`, `latency_samples`
3. Register: `EstimatorRegistry.register(MyEstimator)` in `core/registry.py`
4. Add unit tests in `tests/unit/estimators/test_<name>.py`
5. Verify numerical parity against `v1/PMU/pfebench/` equivalent

---

## Adding a new scenario

1. Create `src/openfreqbench/scenarios/<group>/<name>.py`
2. Subclass `ScenarioBase` as `@dataclass`, implement `build()`
3. Define `tuning_map` for Monte Carlo param injection
4. Register: `ScenarioRegistry.register(MyScenario)` in `core/registry.py`
5. Add unit tests in `tests/unit/scenarios/test_<name>.py`

---

## Artifact layout

```
artifacts/
└── <scenario_id>/
    └── <method_id>/
        └── <method_id>_<YYYYMMDD_HHMMSS>_report.json
```

---

## pyproject.toml notes

- Package source: `src/` layout (`[tool.setuptools.package-dir] "" = "src"`)
- Import as: `from openfreqbench import ...`
- CLI entry: `ofb = "openfreqbench.cli.app:main"`
- Tests use `pythonpath = ["src"]`

---

## DO NOT

- Return fake/hardcoded metric values from any runner
- Add timing inside `BaseEstimator._step()` or `step()`
- Import framework code (pydantic, typer, rich) inside `estimators/`, `scenarios/`, `metrics/`
- Delete or modify `v1/PMU/pfebench/` files without explicit instruction
- Commit binary artifacts (`.pdf`, `.png`, large `.csv`) unless `.gitignore` excludes them
