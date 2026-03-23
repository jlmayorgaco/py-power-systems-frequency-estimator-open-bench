# OpenFreqBench Architecture Audit and Migration Report

---

| Field | Value |
|---|---|
| **Status** | Draft — internal review |
| **Scope** | Full repository audit; consolidation and migration planning |
| **Prepared for** | OpenFreqBench core maintainer |
| **Prepared by** | Principal architecture review (AI-assisted, Claude Sonnet 4.6) |
| **Date** | 2026-03-18 |
| **Repository** | `open-freq-bench` — branch `q1-arch-final` |
| **Basis** | Direct inspection of source files, package configs, test suites, runner logic, artifact schemas, and the `v1/PMU/README.md` research manifesto |

---

## Table of Contents

1. [Executive Assessment](#1-executive-assessment)
2. [Current Architectural Weaknesses](#2-current-architectural-weaknesses)
3. [Intended Benchmark Philosophy (Formalized)](#3-intended-benchmark-philosophy-formalized)
4. [Target Domain Model](#4-target-domain-model)
5. [Canonical Package and CLI Decision](#5-canonical-package-and-cli-decision)
6. [Legacy Concepts to Preserve / Refactor / Archive / Remove](#6-legacy-concepts-to-preserve--refactor--archive--remove)
7. [Recommended Target Architecture](#7-recommended-target-architecture)
8. [Estimator Plugin API Proposal](#8-estimator-plugin-api-proposal)
9. [Incremental Runner and Artifact Persistence Proposal](#9-incremental-runner-and-artifact-persistence-proposal)
10. [Benchmark / Profiling Separation Proposal](#10-benchmark--profiling-separation-proposal)
11. [Statistical Rigor Upgrades](#11-statistical-rigor-upgrades)
12. [Open-Source Readiness Gaps](#12-open-source-readiness-gaps)
13. [Documentation Files to Create](#13-documentation-files-to-create)
14. [Proposed Folder / Package Structure](#14-proposed-folder--package-structure)
15. [Incremental Migration Plan](#15-incremental-migration-plan)
16. [Prioritized Ticket Backlog (P0 / P1 / P2)](#16-prioritized-ticket-backlog-p0--p1--p2)
17. [Conclusion](#17-conclusion)

---

## 1. Executive Assessment

This project is a serious research system for benchmarking dynamic frequency estimators in power systems. The scientific foundations are strong. The architecture has not yet caught up to the scientific ambition.

The repository currently contains three coexisting architectural worlds that were never formally unified:

| World | Location | State |
|---|---|---|
| Old legacy | `estimators/`, `scenarios/`, `evaluation/` | Functional but uses a PMU phasor contract; not packaged |
| **Working research system** | `src/openfreqbench | The real scientific core. Real results, working Monte Carlo, 10+ estimators, 18 scenarios, full test suite. |
| Modern framework skeleton | `ofb/` | Pydantic + Typer + Registry. Clean design, but the benchmark runner is a stub returning `TVE_mean: 0.0`. |

**The single most critical finding:** `ofb/benchmarks/runner.py` silently returns zeroed results. The CLI `ofb bench run` produces no scientific output. The real benchmark logic lives in `src/openfreqbench which is not wired to the CLI.

**The consolidation path is clear:** promote `src/openfreqbench as the canonical domain logic, refactor it into a `src/openfreqbench/` layout, wire the existing `ofb/` CLI and Registry on top of it, and archive the rest. This is an engineering consolidation, not a scientific rewrite.

### 30-Second Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| Canonical Python package | `src/openfreqbench/` | Matches pip name; eliminates `ofb/`-import mismatch |
| CLI command | `ofb` | Short, memorable, Angular-CLI-like |
| Domain logic source of truth | `src/openfreqbench | Only location with working science and real results |
| `ofb/benchmarks/runner.py` | Replace immediately | Returns fake zeros — silent correctness failure |
| `src/openfreqbench fate | Archive to `legacy/` after porting | Keep until numerical parity is verified |
| Worker strategy | `ProcessPoolExecutor` per `(scenario × method)` | Natural granularity; no shared state |
| Benchmark configuration | YAML-driven | Readable, versionable, no Python editing required |
| Plugin discovery | Auto-scan + entry points | Built-in estimators auto-discovered; external plugins via entry points |

---

## 2. Current Architectural Weaknesses

### 2.1 Package Identity and Import Confusion

- `pyproject.toml` declares `name = "openfreqbench"` (pip install name), but `setuptools.packages.find` includes only `ofb*`, so the import is `import ofb`. These two names do not match.
- A third name, `openfreqbench`, exists in `v1/PMU/pyproject.toml` as a separate installable package.
- Three package identities (`openfreqbench`, `ofb`, `openfreqbench`) for one project. Contributors cannot determine the canonical import path.

### 2.2 Runner Architecture

- **`ofb/benchmarks/runner.py` is a stub.** The `run_benchmark()` function returns hardcoded zero values for all metrics. Running `ofb bench run` produces no scientific output and gives no error.
- The real Monte Carlo pipeline (`run_mini_mc`, `run_statistical_analysis`, `save_artifacts`, `compare_methods`) lives in `src/openfreqbench and is inaccessible from the CLI.
- Estimator and scenario lists are hardcoded in `src/openfreqbench Adding a new estimator requires editing the runner script.
- No cache or resume mechanism. Every run starts from scratch. The artifact path uses `ts_now()` (a wall-clock timestamp), so each run creates a new directory instead of reusing or skipping existing results.
- No parallelism. All `scenario × method × repetition` loops are sequential.

### 2.3 Testing

- `tests/` at the repo root contains 3 files, 2 of which are empty stubs.
- The real test suite (13 estimator tests, 18 scenario tests, metrics tests) lives in `src/openfreqbench and is not discovered by `pytest` when run from the repo root.
- No smoke-test subset. A contributor cannot run a quick validation in under a minute.

### 2.4 Repo Hygiene

- `v1/PMU/artifacts/mini_mc_compare/` — hundreds of CSV, JSON, PNG, and PDF generated files — are tracked in git. This is wrong; generated artifacts should be gitignored or stored in a data release.
- `openfreqbench.egg-info/` (a build artifact) is tracked in git.
- The branch `q1-arch-final` deleted `.github/workflows/ci.yml`, all issue templates, the labeler config, and the release config. There is no CI at this time.
- A large `_____TEMP____/` tree is present in the git diff as deletions, polluting history.
- Stub scenario files in `scenarios/s2_ieee13/` through `s6_real_csv/` exist without implementations, creating a false impression of coverage.

### 2.5 Coupling and Responsibility Problems

- **Estimator self-times its own execution.** In `openfreqbench/estimators/base.py`, `step()` wraps `_step()` with `perf_counter_ns`. Timing is a framework responsibility, not an estimator responsibility.
- **Tuning (GSO) lives inside `BaseEstimator.optimize()`.** An estimator should not own its own tuning logic. This violates the Single Responsibility Principle.
- **Plotting is mixed with orchestration.** `validate_estimator.py` calls `matplotlib` directly. Plots should be generated in an isolated `plotting/` layer.
- **Three incompatible estimator contracts coexist:**
  - `estimators/base.py`: `update(measures: PMU_Input) → PMU_Output` (phasor contract)
  - `src/openfreqbench `_step(v_sample: float) → float` (the correct one)
  - `ofb/core/estimators_api.py`: `update(t, x) → dict{t, seq, fhat, meta}` (decorator-based)

### 2.6 Documentation and Open-Source Readiness

- `docs/` contains only empty Markdown stubs.
- `CHANGELOG.md` is empty.
- `README.md` describes features (`ofb bench run`, full benchmark matrix) that do not work as described.
- No `CONTRIBUTING.md` explaining how to add an estimator or scenario.
- No `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, or `RESULT_SCHEMA.md`.
- No benchmark configuration format documented.

---

## 3. Intended Benchmark Philosophy (Formalized)

The `v1/PMU/README.md` manifesto and the artifact structure in `v1/PMU/artifacts/` together define the intended benchmark philosophy clearly. This section formalizes it.

### 3.1 Natural Execution Unit

The natural unit of work in this framework is:

```
scenario × method × repetition (seed)
```

Outputs at this level: one raw voltage trace, one estimated frequency trace, one timing profile, one run metadata record.

These are aggregated upward through a four-level hierarchy:

```
scenario × method × repetition    →  raw trace + run metadata
scenario × method                 →  MC summary + statistics
scenario (all methods)            →  ranked comparison
suite (all scenarios)             →  cross-scenario aggregation + Pareto
```

### 3.2 Benchmark Modes

#### Mode 1 — Best-Case Tuned Per-Scenario (Canonical, Existing)

Each estimator is grid-searched on a held-out calibration waveform (fixed seed, e.g., `seed=999`) for the specific scenario it will be evaluated on. Monte Carlo repetitions then run with the best-found parameters and variable random seeds. This measures **oracle best-case performance** for each (scenario, method) pair.

This mode is intentional and must be preserved. It answers: *What is the best this algorithm can achieve on this scenario type under optimal tuning?*

#### Mode 2 — Transfer / Generalization Benchmark (Missing, Needed)

An estimator tuned on one scenario family (e.g., G1 nominal) is evaluated on a different family (e.g., G3 complex events) without re-tuning. This measures **generalization performance**.

This does not replace Mode 1. Both modes coexist with explicit labeling and answer different scientific questions.

#### Mode 3 — Sensitivity / Robustness Benchmark (Partial)

Structured perturbation of scenario parameters around the best-found configuration. Answers: *How fragile is the best configuration to parameter variation?*

#### Mode 4 — Resource / Profiling Benchmark (Partial)

Framework-measured CPU time per sample, RAM footprint, structural latency, and empirical output delay. Measured externally by the framework, not self-reported by the estimator.

#### Mode 5 — Standard Compliance Benchmark (Planned)

Pass/fail compliance against IEEE C37.118 or IEC 60255-118 frequency/ROCOF envelopes, with Trip-Risk Duration (TRD) as the compliance metric.

### 3.3 Incremental Persistence Philosophy

The framework must support incremental execution by scenario and method. A researcher must be able to:

- Run G1 with method ZC, persist results, continue to method EKF without rerunning ZC.
- Move to G2 after G1 is complete, without touching G1 artifacts.
- Resume an interrupted benchmark run from the last valid artifact.
- Recompute statistics and reports from saved raw artifacts without re-running simulations.

**Forcing full reruns when valid artifacts already exist is an architectural defect, not an acceptable behavior.**

---

## 4. Target Domain Model

The following tables define the conceptual entities the framework needs, their responsibilities, and the constraints on their knowledge boundaries.

### Layer 0 — Core Primitives

| Entity | Responsibility | Belongs To | Must Not Know About |
|---|---|---|---|
| `BaseEstimator` | `__init__(config)`, `_step(v) → float`, `reset()`, `latency_samples` | `estimators/_base.py` | Scenarios, metrics, tuning, plotting, runners |
| `EstimatorOutput` | Typed return: `{f_hat, rocof_hat?, valid, latency_s}` | `estimators/_base.py` | Framework internals |
| `EstimatorMeta` | `name, family, description, complexity_class, supports_rocof` | `estimators/_base.py` | Runtime state |
| `TuningParam` | `{name, type, values/range, default, description}` | `estimators/_base.py` | Estimator implementation |
| `ScenarioBase` | `build(seed) → ScenarioOutput`, `scenario_id`, `tuning_map` | `scenarios/_base.py` | Estimators, metrics, runners |
| `ScenarioState` | `t, v, f_true, phi, A, fs_hz, f_nom_hz, seed` | `scenarios/_base.py` | Estimators |
| `ScenarioOutput` | Wraps `ScenarioState` + `scenario_id` + `schema` | `scenarios/_base.py` | Runners |
| `NoiseModel` | Stateless: `apply(signal, rng) → signal` | `scenarios/_base.py` | Scenarios |

### Layer 1 — Configuration

| Entity | Responsibility | Belongs To | Must Not Know About |
|---|---|---|---|
| `BenchmarkConfig` | Top-level YAML: scenarios, methods, MC config, tuning config, metrics, mode | `core/config.py` | Estimator internals |
| `MonteCarloConfig` | `n_runs, seeds_strategy, warmup_samples` | `core/config.py` | Estimators |
| `TuningConfig` | `metric, strategy, max_seconds, stride, early_stop` | `core/config.py` | Estimators |
| `TransferConfig` | `train_scenario_ids, eval_scenario_ids` | `core/config.py` | Runner internals |
| `ProfilingConfig` | `measure_time, measure_memory, n_warmup_iters` | `core/config.py` | Estimators |

### Layer 2 — Execution

| Entity | Responsibility | Belongs To | Must Not Know About |
|---|---|---|---|
| `TraceRunner` | One `(scenario, method, seed)` → raw trace + metadata | `runners/trace_runner.py` | Aggregation, stats, plots |
| `TuningRunner` | Calibration signal → `{best_params, best_score, history}` | `tuning/grid_search.py` | MC, stats, plots |
| `ScenarioMethodRunner` | Aggregate N repetitions → metrics summary | `runners/scenario_method_runner.py` | Plots, reports |
| `ScenarioRunner` | Compare all methods within one scenario | `runners/scenario_runner.py` | Suite-level logic |
| `SuiteRunner` | Aggregate across all scenarios | `runners/suite_runner.py` | Estimator internals |
| `ArtifactStore` | Persist/load artifacts with identity-based paths | `io/artifact_store.py` | Runners |
| `ArtifactIdentity` | Deterministic cache key for `(scenario, method, config, version, seed)` | `core/run_identity.py` | All domain logic |

### Layer 3 — Analysis

| Entity | Responsibility | Belongs To | Must Not Know About |
|---|---|---|---|
| `MetricSet` | Collection of scalar metrics from `(f_est, f_true)` | `metrics/` | Estimators, runners |
| `TimingProfile` | `init_us, per_sample_us_p50/p95/p99, per_trace_ms` | `profiling/timing.py` | Estimators |
| `MemoryProfile` | `rss_bytes, pss_bytes, tracemalloc_bytes` | `profiling/memory.py` | Estimators |
| `LatencyProfile` | `structural_samples, empirical_delay_ms` | `profiling/latency.py` | Estimators |
| `StatisticalReport` | CI, Welch t-test, bootstrap, ANOVA, effect sizes | `stats/` | Runners, plots |

### Layer 4 — Artifacts

| Entity | Responsibility | Belongs To | Must Not Know About |
|---|---|---|---|
| `RawTraceArtifact` | Per-run CSV: `sample_idx, t, v, f_true, f_est, error, valid, timing_us` | `io/` | Analysis logic |
| `RunMetadataArtifact` | Per-run JSON: IDs, config hashes, seed, git hash, version | `io/` | Domain logic |
| `ScenarioMethodSummary` | Aggregated metrics + stats over MC runs | `io/` | Plotting |
| `ScenarioComparisonArtifact` | Multi-method comparison table per scenario | `io/` | Suite logic |
| `SuiteSummaryArtifact` | Cross-scenario aggregation | `io/` | External consumers |

---

## 5. Canonical Package and CLI Decision

### Decision

| Dimension | Choice |
|---|---|
| **Installable package name** | `openfreqbench` (already in `pyproject.toml`) |
| **Python import namespace** | `from openfreqbench import ...` (replaces `import ofb`) |
| **CLI command** | `ofb` (exposed via console script entry point) |
| **Separate `ofb` Python package** | No — `ofb/` is renamed to `openfreqbench/`, no separate package needed |
| **`openfreqbench` package** | Archived to `legacy/`, not distributed |

### Justification

- `pyproject.toml` already declares `name = "openfreqbench"`. The pip installable name should match the Python import name.
- `openfreqbench` is unambiguous, descriptive, and searchable on PyPI. `ofb` as an import namespace is too short and collision-prone.
- `ofb` as a CLI verb is correct: short, memorable, and consistent with the Angular CLI pattern already established in the codebase.
- A separate `ofb` Python package is unnecessary — `ofb` as a CLI entry point is exposed via `[project.scripts]`, which maps `ofb` → `openfreqbench.cli.app:main`.

### Console Scripts

```toml
[project.scripts]
ofb     = "openfreqbench.cli.app:main"
ofb-new = "openfreqbench.cli.scaffold:main"
```

### Migration Consequence

All source files currently under `ofb/` are moved to `src/openfreqbench/`. All internal imports of the form `from ofb.X import Y` become `from openfreqbench.X import Y`. A compatibility shim is provided during the transition period:

```python
# src/openfreqbench/compat/ofb_shim.py
import warnings
warnings.warn(
    "Importing from 'ofb.*' is deprecated. Use 'openfreqbench.*'.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## 6. Legacy Concepts to Preserve / Refactor / Archive / Remove

| Legacy Concept | Decision | Why | Migration Note |
|---|---|---|---|
| `openfreqbench/estimators/base.py` — `BaseEstimator` with `_step()`, `reset()`, `step()` | **Preserve with refactor** | Best API. Clean single-input interface. Used by 10+ implementations. | Remove self-timing from `step()`; remove `optimize()` method |
| `openfreqbench/scenarios/base.py` — `ScenarioBase`, `ScenarioState`, `ScenarioOutput`, `ScenarioModifiersMixin` | **Preserve** | Production-quality. Rich schema, noise mixin, `tuning_map` for MC. | Fix imports only |
| `openfreqbench/scenarios/G*.py` — all 18 scenarios | **Preserve** | Real science. Working physics and noise models. | Port to `src/openfreqbench/scenarios/gN/` |
| `openfreqbench/estimators/e1_zc.py` through `e10_nr.py` | **Preserve** | 10 working estimators with real implementations. | Port to `src/openfreqbench/estimators/` |
| `openfreqbench/runners/validate_estimator.py` — `run_mini_mc()`, `run_statistical_analysis()`, `save_artifacts()`, `compare_methods()` | **Preserve with refactor** | The real orchestration logic. | Extract into `runners/`, `stats/`, `io/` modules |
| `openfreqbench/tests/` — all 13+ estimator tests and 18+ scenario tests | **Preserve** | Port as migration anchors and regression tests. | Move to `tests/unit/` |
| `openfreqbench/metrics/metrics.py` — `compute_metrics()`, `aggregate_monte_carlo()` | **Preserve** | Solid. | Move to `src/openfreqbench/metrics/` |
| `BaseEstimator.optimize()` (GSO embedded in estimator) | **Preserve with refactor** | Correct algorithm; wrong location. | Extract to `tuning/grid_search.py:TuningRunner` |
| `BaseEstimator.step()` timing via `perf_counter_ns` | **Preserve with refactor** | Correct mechanism; wrong owner. | Move to `profiling/timing.py:TimingHarness` |
| `ofb/core/registry.py` | **Preserve** | Clean registry pattern. | Keep, wire to real estimators |
| `ofb/config/models.py` — Pydantic config models | **Preserve with refactor** | Clean structure. | Extend with `MonteCarloConfig`, `TuningConfig`, `ProfilingConfig` |
| `ofb/runtime/profiling.py` — `MemoryMeter` | **Preserve** | Well-designed RSS/PSS/tracemalloc measurement. | Move to `profiling/memory.py`; wire into `TraceRunner` |
| `ofb/cli/` — Typer CLI structure | **Preserve** | Correct architecture. | Wire commands to real logic |
| `estimators/base.py` — PMU phasor contract | **Archive** | Superseded by openfreqbench base. | Move to `legacy/` |
| `estimators/basic/ipdft.py`, `estimators/basic/zcd/` | **Archive** | Superseded by openfreqbench implementations. | Move to `legacy/`; port algorithm if needed |
| `scenarios/s0_sin_wave/`, `scenarios/s1_synthetic/` | **Archive** | Raw-tuple-returning generators; superseded by `ScenarioBase`. | Port physics to proper `ScenarioBase` subclasses |
| `scenarios/s2_ieee13/` through `s6_real_csv/` | **Archive** | Stub files without implementations. | Move to `legacy/`; implement properly in future |
| `pipelines/run_benchmark.py` | **Archive** | Legacy script runner. | Move to `legacy/` |
| `ofb/benchmarks/runner.py` stub | **Remove** | Returns fake zeroed results. | Replace with real `SuiteRunner` |
| `core/pmu/` at root | **Remove** | Empty stub files. | Delete |
| `_____TEMP____/` traces in git history | **Remove** | Pollution. | `git rm --cached` |
| `v1/PMU/artifacts/` tracked in git | **Remove from tracking** | Generated data should not be in source control. | Gitignore; distribute as release assets |

---

## 7. Recommended Target Architecture

### 7.1 Package Tree

```
src/
└── openfreqbench/
    ├── __init__.py            # version only; minimal re-exports
    ├── _version.py            # __version__ = "0.2.0"
    │
    ├── cli/                   # CLI layer — no domain logic
    │   ├── app.py             # Typer root: ofb command
    │   ├── scaffold.py        # ofb new estimator/scenario/metric
    │   ├── commands/
    │   │   ├── run.py         # ofb run --config benchmark.yaml
    │   │   ├── analyze.py     # ofb analyze scenario G1
    │   │   ├── status.py      # ofb status --config benchmark.yaml
    │   │   ├── doctor.py      # ofb doctor
    │   │   └── registry.py    # ofb list estimators / scenarios
    │   └── _output.py         # Rich tables, banners, progress
    │
    ├── core/                  # Framework internals — no estimators, no scenarios
    │   ├── registry.py        # discover, register, resolve
    │   ├── config.py          # Pydantic: BenchmarkConfig, MonteCarloConfig, TuningConfig
    │   ├── run_identity.py    # ArtifactIdentity, cache key, validity check
    │   └── errors.py
    │
    ├── estimators/            # Estimator plugins — one file per estimator
    │   ├── _base.py           # BaseEstimator, TuningParam, EstimatorMeta, EstimatorOutput
    │   ├── zc.py              # ZeroCrossing
    │   ├── izc.py             # InterpolatedZeroCrossing
    │   ├── pm.py              # PeriodMeasurement
    │   ├── if_dphi.py         # InstantaneousFrequencyPhaseIncrement
    │   ├── zc_ma.py           # ZeroCrossingMovingAverage
    │   ├── mwls.py            # MovingWindowLeastSquares
    │   ├── rls.py             # RecursiveLeastSquares
    │   ├── wls.py             # WindowedLeastSquares
    │   ├── ar.py              # Autoregressive
    │   ├── prony.py           # Prony
    │   └── nr.py              # NewtonRaphson
    │
    ├── scenarios/             # Scenario plugins — one file per scenario
    │   ├── _base.py           # ScenarioBase, ScenarioState, ScenarioOutput, Mixin
    │   ├── catalog.yaml
    │   ├── g1/                # Group 1: Baseline nominal
    │   ├── g2/                # Group 2: Voltage and frequency events
    │   ├── g3/                # Group 3: Modulation, outliers, complex distortions
    │   └── g4/                # Group 4: Composite and extreme events
    │
    ├── metrics/               # Pure metric functions — stateless
    │   ├── _base.py           # MetricConfig, MetricResult, MetricSet
    │   ├── frequency.py       # RMSE_Hz, MAE_Hz, FE_Max, FE_mHz
    │   ├── rocof.py           # ROCOF_RMSE, ROCOF_Max
    │   ├── compliance.py      # IEEE C37.118 pass/fail, TRD
    │   └── aggregate.py       # aggregate_monte_carlo(), compute_metrics()
    │
    ├── profiling/             # Resource measurement — framework-owned
    │   ├── timing.py          # TimingHarness: wraps _step() with perf_counter_ns
    │   ├── memory.py          # MemoryMeter: RSS, PSS, tracemalloc
    │   └── latency.py         # structural_latency(), empirical_output_delay()
    │
    ├── tuning/                # Tuning strategies — external to estimators
    │   ├── _base.py           # TuningRunner protocol
    │   └── grid_search.py     # GSO: grid search, early stop, stride, time budget
    │
    ├── runners/               # Benchmark execution hierarchy
    │   ├── trace_runner.py    # (scenario, method, seed) → RawTraceArtifact
    │   ├── scenario_method_runner.py  # MC aggregation
    │   ├── scenario_runner.py # Multi-method comparison within one scenario
    │   ├── suite_runner.py    # Cross-scenario aggregation
    │   └── worker_pool.py     # ProcessPoolExecutor dispatch
    │
    ├── stats/                 # Statistical analysis — pure functions
    │   ├── monte_carlo.py     # bootstrap CI, aggregate_monte_carlo
    │   ├── hypothesis.py      # Welch t-test, ANOVA, Mann-Whitney, Bonferroni
    │   ├── effects.py         # Cohen's d, rank-biserial correlation
    │   └── ranking.py         # Spearman/Kendall ranking stability
    │
    ├── reports/               # Artifact synthesis
    │   ├── json_report.py
    │   ├── csv_report.py
    │   └── comparison.py      # compare_methods(), multi-method ranking
    │
    ├── plotting/              # Visualization — isolated from scientific logic
    │   ├── timeseries.py      # frequency trace with BrokenAxes zoom
    │   ├── boxplot.py
    │   ├── pareto.py          # Pareto frontier (precision vs latency)
    │   ├── heatmap.py
    │   └── violin.py
    │
    ├── io/                    # Artifact persistence — I/O layer only
    │   ├── csv_io.py
    │   ├── json_io.py         # NumpyEncoder, read/write report JSON
    │   ├── parquet_io.py      # optional
    │   └── artifact_store.py  # ArtifactStore with identity-based paths
    │
    └── compat/                # Backward compatibility shims (temporary)
        └── openfreqbench.py        # import openfreqbench.X → openfreqbench.X + DeprecationWarning
```

### 7.2 Dependency Direction Rules

These rules must never be violated. They prevent circular imports and keep the scientific core framework-independent.

```
cli/          → core/, runners/, io/, plotting/, reports/
runners/      → estimators/, scenarios/, metrics/, profiling/, tuning/, stats/, io/
tuning/       → estimators/, metrics/
metrics/      → (pure functions; no framework dependencies)
profiling/    → (pure functions + psutil; no framework dependencies)
stats/        → (pure functions + numpy/scipy; no framework dependencies)
plotting/     → (matplotlib only; receives data structures; no runners/metrics imports)
io/           → (pathlib, pandas, json only)
estimators/   → (numpy only; no framework imports)
scenarios/    → (numpy only; no framework imports)
```

### 7.3 Layer Boundaries

The key principle is that **scientific core code (estimators and scenarios) must have zero awareness of the framework**. An estimator implemented today must be runnable standalone without importing anything from `openfreqbench`. The framework wraps the estimator; the estimator does not reach into the framework.

---

## 8. Estimator Plugin API Proposal

### 8.1 Contributor-Facing API (Minimal)

A contributor should be able to add a working estimator with a single file:

```python
# src/openfreqbench/estimators/my_ekf.py

from openfreqbench.estimators._base import BaseEstimator, TuningParam, EstimatorOutput
from typing import List
import numpy as np

class MyEKF(BaseEstimator):
    """Extended Kalman Filter frequency estimator."""

    NAME   = "EKF"
    FAMILY = "state_space"
    COMPLEXITY = "O(n)"   # optional: per-sample complexity class

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Initialize all internal state from self._params here.
        self._Q = self._params["process_noise"]
        self._R = self._params["measurement_noise"]
        # Kalman matrices, state vector, etc.

    def _step(self, v: float) -> float:
        """
        Core estimation logic. Pure math.
        Receives one voltage sample. Returns f_hat in Hz.
        All state is in self._XXX, initialized in __init__ and reset().
        """
        # EKF predict + update steps
        return f_hat  # float

    def reset(self) -> None:
        """Reinitialize all internal buffers to t=0."""
        self._x = np.array([60.0, 0.0])
        self._P = np.eye(2) * 1e-3

    @property
    def latency_samples(self) -> int:
        """Structural latency in samples (for causal alignment in metrics)."""
        return 0

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """Declare the hyperparameter search space for grid search tuning."""
        return [
            TuningParam("process_noise", 1e-4, "float",
                        range=(1e-6, 1e-2, 10), description="Q diagonal"),
            TuningParam("measurement_noise", 1e-3, "float",
                        range=(1e-5, 1e-1, 10), description="R value"),
        ]
```

### 8.2 Required vs Optional Methods

| Method | Required | Notes |
|---|---|---|
| `__init__(params: dict \| None)` | Yes | Initialize state from `self._params` |
| `_step(v: float) → float` | Yes | Core algorithm. Framework wraps with `TimingHarness`. |
| `reset() → None` | Yes | Restore t=0 state completely |
| `latency_samples → int` (property) | Yes | For causal alignment before metric computation |
| `tuning_ranges() → List[TuningParam]` | No | Return `[]` if estimator is not tunable |
| `supports_rocof_output() → bool` | No | If `True`, framework collects ROCOF output |
| `get_state() → dict` | No | Expose internal state for debugging |
| `metadata() → dict` | No | Extra metadata for reports (paper citation, complexity note) |

### 8.3 Timing Responsibility

In the current openfreqbench codebase, `BaseEstimator.step()` wraps `_step()` with `perf_counter_ns`. This must move to the framework:

```python
# profiling/timing.py
class TimingHarness:
    def run_timed_trace(
        self, estimator: BaseEstimator, v_array: np.ndarray, n_warmup: int = 3
    ) -> TimingResult:
        # Warmup — not measured
        for v in v_array[:n_warmup]:
            estimator.step(v)
        estimator.reset()

        per_sample_us = np.empty(len(v_array))
        f_hat_arr = np.empty(len(v_array))

        for i, v in enumerate(v_array):
            t0 = time.perf_counter_ns()
            f_hat_arr[i] = estimator.step(v)
            per_sample_us[i] = (time.perf_counter_ns() - t0) / 1000.0

        return TimingResult(
            f_hat=f_hat_arr,
            per_sample_us=per_sample_us,
            p50_us=float(np.percentile(per_sample_us, 50)),
            p95_us=float(np.percentile(per_sample_us, 95)),
            p99_us=float(np.percentile(per_sample_us, 99)),
            total_trace_ms=per_sample_us.sum() / 1000.0,
        )
```

### 8.4 Scaffolding

```bash
ofb new estimator --name my_ekf --family state_space
```

Generates:
- `src/openfreqbench/estimators/my_ekf.py` (populated from template)
- `tests/unit/estimators/test_my_ekf.py` (populated test template)

### 8.5 Minimum Tests per Estimator

Every estimator must pass:

1. **Convergence on pure sine:** RMSE < 0.05 Hz on `G1_E1_Pure_60Hz` after warmup.
2. **No NaN output:** `np.all(np.isfinite(estimator.run(v_pure_sine)))`.
3. **Determinism:** two calls with the same input and `reset()` between them return identical arrays.
4. **Reset clears state:** behavior after `reset()` matches behavior on a fresh instance.

### 8.6 External Plugin Support

For third-party community contributions:

```toml
# A contributor's pyproject.toml
[project.entry-points."openfreqbench.estimators"]
my_estimator = "my_package.estimators.my_ekf:MyEKF"
```

The Registry scans `importlib.metadata.entry_points` at startup and auto-registers external estimators.

---

## 9. Incremental Runner and Artifact Persistence Proposal

### 9.1 Four-Level Runner Hierarchy

```
SuiteRunner
  └── ScenarioRunner                    (per scenario)
        └── ScenarioMethodRunner        (per scenario × method)
              └── TraceRunner           (per scenario × method × seed)
```

### 9.2 Level 1 — TraceRunner

**Responsibility:** Atomic execution of one `(scenario, method, seed)` triple.

**Steps:**
1. `scenario.build(seed)` → `ScenarioOutput`
2. Instantiate `estimator_cls(params)`; call `estimator.reset()`
3. For each sample: `TimingHarness` calls `estimator.step(v)`; records `f_hat` and `timing_us`
4. `MemoryMeter` measures RSS delta post-warmup
5. Compute causal alignment offset from `estimator.latency_samples`
6. Return `TraceResult` with arrays + `TimingProfile`

**Output artifacts:**
```
artifacts/<scenario_id>/<method_id>/<run_id>/
  trace.csv       # sample_idx, t, v, f_true, f_est, error_hz, valid, timing_us
  metadata.json   # run_id, scenario_id, scenario_config_hash, method_id,
                  # method_params_hash, seed, policy_version, schema_version,
                  # git_hash, timestamp, status
```

### 9.3 Artifact Identity and Cache Key

```python
@dataclasses.dataclass(frozen=True)
class ArtifactIdentity:
    scenario_id: str
    scenario_config_hash: str   # SHA-256 of scenario params JSON
    method_id: str
    method_config_hash: str     # SHA-256 of estimator params JSON
    seed: int
    policy_version: str         # e.g., "v1.0" — benchmark protocol version
    schema_version: str         # e.g., "v1.0" — artifact format version

    @property
    def cache_key(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

A run is considered **valid** and **skippable** when:
- `metadata.json` exists at the expected path
- `metadata.json` contains matching `cache_key`
- `trace.csv` exists and is non-empty
- `status` field is `"completed"`

### 9.4 Level 2 — ScenarioMethodRunner

**Responsibility:** Aggregate N Monte Carlo repetitions for one `(scenario, method)` pair.

**Resume logic:**
```python
for seed in range(mc_config.n_runs):
    identity = ArtifactIdentity(scenario_id, ..., seed=seed, ...)
    if resume and store.exists_valid(identity):
        # load existing trace from disk — no re-simulation
        continue
    result = trace_runner.run(scenario, estimator_cls, params, seed)
    store.save_trace(identity, result)

# After all seeds: aggregate
summaries = [store.load_trace(identity) for seed in range(mc_config.n_runs)]
metrics = aggregate_monte_carlo(summaries)
stats_report = run_statistical_analysis(metrics)
store.save_scenario_method_summary(scenario_id, method_id, metrics, stats_report)
```

**Output artifacts:**
```
artifacts/<scenario_id>/<method_id>/
  summary.json         # RMSE_mean, RMSE_std, CI_95, MAE, CVaR_95, profiling summary
  runs_long.csv        # all seeds stacked: seed, sample_idx, t, f_true, f_est, error
  tuning_history.csv   # GSO iterations: params, RMSE score
```

### 9.5 Level 3 — ScenarioRunner

**Responsibility:** Tuning + MC for each method; comparison across all methods within one scenario.

**Steps per method:**
1. Generate calibration waveform with fixed `CALIB_SEED` (e.g., 999)
2. `TuningRunner.run(estimator_cls, v_cal, f_cal)` → `TuningResult`
3. `ScenarioMethodRunner.run(...)` using best-found params

After all methods:

4. `compare_methods(all_summaries)` → ranked table, pairwise Welch t-tests, bootstrap CI
5. Save scenario-level comparison artifacts

**Output artifacts:**
```
artifacts/<scenario_id>/
  comparison_summary.json   # ranked methods, pairwise p-values, effect sizes
  comparison_runs_long.csv  # all methods and seeds combined
  comparison_plot.pdf
```

### 9.6 Level 4 — SuiteRunner

**Responsibility:** Dispatch scenario runners (optionally in parallel) and aggregate cross-scenario results.

**Parallelism:**
```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {
        pool.submit(_run_scenario_method, scenario_id, method_id, config, store_path):
            (scenario_id, method_id)
        for scenario_id in config.scenario_ids
        for method_id in config.method_ids
    }
    for future in as_completed(futures):
        scenario_id, method_id = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            logger.error("Failed %s × %s: %s", scenario_id, method_id, exc)
```

**Worker granularity:** `(scenario × method)` is the natural unit. Each worker has its own artifact paths and no shared state, avoiding race conditions.

**Output artifacts:**
```
artifacts/
  suite_summary.json       # cross-scenario aggregation, Pareto data
  suite_summary_wide.csv   # methods as columns, scenarios as rows, RMSE values
  pareto_frontier.pdf
  ranking_stability.json
```

### 9.7 YAML Configuration

```yaml
# benchmark.yaml
benchmark:
  name: "baseline_freq_v1"
  policy_version: "v1.0"
  schema_version: "v1.0"
  mode: "best_case_tuned"   # or: transfer, sensitivity, profiling

monte_carlo:
  n_runs: 100
  seeds_strategy: "sequential"   # seeds 0..99
  warmup_samples: 10

tuning:
  metric: "RMSE_HZ"
  strategy: "grid"
  max_seconds: 30.0
  stride: 1
  early_stop_rmse: 0.001
  patience: 5

profiling:
  measure_time: true
  measure_memory: true
  n_warmup_iters: 3

scenarios:
  - id: "G1_E1_Pure_60Hz"
    params: {fs_hz: 10000.0, T_s: 5.0, f_nom_hz: 60.0}
  - id: "G2_E7_Freq_Step_60_to_55"
    params: {fs_hz: 10000.0, T_s: 5.0}

methods:
  - id: "ZeroCrossing"
    params: {}
  - id: "EKF"
    params: {}   # empty = use tuning to find best params

targets:
  RMSE_HZ_TARGET: 0.01
  IEEE_PASS_RATE_TARGET: 0.95

workers: 4
resume: true
output_dir: "./artifacts/baseline_freq_v1"
```

### 9.8 CLI Commands

```bash
ofb run --config benchmark.yaml
ofb run scenario G1_E1_Pure_60Hz --method ZeroCrossing --seeds 0:5
ofb status --config benchmark.yaml
ofb resume --config benchmark.yaml
ofb analyze scenario G1_E1_Pure_60Hz
ofb analyze suite baseline_freq_v1
```

---

## 10. Benchmark / Profiling Separation Proposal

### 10.1 Current Violations

| Violation | Location | Severity |
|---|---|---|
| Estimator self-times its own `step()` | `openfreqbench/estimators/base.py:256–264` | High — estimator should not own timing |
| Tuning (GSO) embedded in `BaseEstimator.optimize()` | `openfreqbench/estimators/base.py:162–227` | Medium — Single Responsibility Principle violation |
| Matplotlib plot generation inside `validate_estimator.py` | `openfreqbench/runners/validate_estimator.py` | Medium — plotting mixed with orchestration |
| Bootstrap CI computed inside `validate_estimator.py` | Same file | Low — should live in `stats/` |
| `save_artifacts()` handles both JSON building and file I/O | Same file | Low — minor SRP violation |

### 10.2 Target Separation

```
ESTIMATOR (_step only, pure math)
    │
    ▼ called by
TRACE RUNNER
    ├── calls scenario.build(seed)          → ScenarioOutput
    ├── calls profiling.TimingHarness       → wraps each _step() externally
    ├── calls profiling.MemoryMeter         → RSS before/after
    ├── collects f_est[], timing_us[] arrays
    └── calls io.save_trace()               → trace.csv + metadata.json
    │
    ▼ results passed to
TUNING RUNNER
    ├── receives calibration (v_cal, f_cal)
    ├── iterates TuningParam grid
    ├── calls estimator.set_params() + short TraceRunner runs
    └── returns {best_params, best_score, history}
    │
    ▼ best_params used in
SCENARIO METHOD RUNNER (MC aggregation)
    ├── calls TraceRunner N times with N seeds
    ├── passes traces to metrics.compute_metrics()
    ├── passes metric arrays to stats.aggregate_monte_carlo()
    └── passes aggregated stats to io.save_summary()
    │
    ▼ summaries aggregated by
SCENARIO RUNNER → SUITE RUNNER
    ├── calls reports.comparison.compare_methods()
    └── calls plotting.* for figures (fire-and-forget, no scientific logic)
```

### 10.3 Profiling Architecture

**Timing** (`profiling/timing.py` — see Section 8.3 for full code)
- Wraps `estimator.step()` externally using `perf_counter_ns`.
- Reports `p50_us`, `p95_us`, `p99_us`, and `total_trace_ms`.

**Memory** (`profiling/memory.py` — promoted from `ofb/runtime/profiling.py`)
- Measures RSS delta and tracemalloc peak for a fully initialized estimator.
- Called once per `(scenario × method)`, not per sample.

**Latency** (`profiling/latency.py`)
- Structural latency: read from `estimator.latency_samples` (estimator-declared).
- Empirical output delay: cross-correlation of `f_est` and `f_true`.

### 10.4 What Belongs Inside vs Outside the Estimator

| Concern | Belongs Inside Estimator | Belongs Outside Estimator |
|---|---|---|
| Core estimation math | Yes — `_step()` | — |
| Internal state buffers | Yes — `__init__`, `reset()` | — |
| Latency declaration | Yes — `latency_samples` property | — |
| Tuning parameter declaration | Yes — `tuning_ranges()` | — |
| Timing measurement | — | `TimingHarness` in `profiling/` |
| Memory measurement | — | `MemoryMeter` in `profiling/` |
| Tuning / grid search execution | — | `TuningRunner` in `tuning/` |
| Signal generation | — | `ScenarioBase.build()` |
| Metric computation | — | `metrics/` |
| Artifact saving | — | `io/` |
| Statistical analysis | — | `stats/` |
| Plotting | — | `plotting/` |

---

## 11. Statistical Rigor Upgrades

### 11.1 Current State vs Recommended

| Check | Current Status | Recommendation | Priority |
|---|---|---|---|
| Monte Carlo repetitions | Implemented (`run_mini_mc()`) | Set default n_runs = 100; enforce minimum n_runs ≥ 30 | P0 |
| RMSE, MAE metrics | Implemented | Keep | — |
| Bootstrap CI | Implemented | Keep | — |
| Welch t-test (pairwise) | Implemented (scipy optional) | Make required; pin scipy | P0 |
| Bonferroni / Holm correction | Missing | Add; report corrected p-values alongside raw | P1 |
| Effect sizes (Cohen's d) | Missing | Add to every method-pair comparison | P1 |
| Paired statistical tests | Not verified | Use Wilcoxon signed-rank when same seeds are used for both methods | P1 |
| CVaR / p95 / worst-case | Missing | Add `cvar_95_hz` to all scenario-method summaries | P1 |
| KDE / distribution shape | Missing | Add to suite-level violin plots | P2 |
| N-way ANOVA (method × scenario interaction) | Missing | Add at suite-level | P2 |
| Ranking stability (Spearman/Kendall) | Missing | Add to suite summary | P2 |
| Confidence intervals on timing | Missing | Report p95 timing CI across MC runs | P1 |
| Fixed seed documentation | Partial (calibration seed=999 hardcoded) | Codify seed strategy in `MonteCarloConfig`; record in metadata | P0 |

### 11.2 Multiple Comparison Correction

When comparing K methods across S scenarios, the probability of at least one false positive grows with K. Apply Bonferroni or Holm-Bonferroni correction to pairwise p-values:

```python
# stats/hypothesis.py
def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    n = len(pvalues)
    corrected = np.minimum(pvalues * n, 1.0)
    return {
        "corrected_pvalues": corrected.tolist(),
        "reject": (corrected < alpha).tolist(),
        "alpha_corrected": alpha / n,
    }
```

### 11.3 Effect Sizes

P-values indicate whether a difference exists; effect sizes indicate how large it is:

```python
# stats/effects.py
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0
```

### 11.4 Tail-Risk Metric (CVaR-95)

RMSE underweights catastrophic tail errors that cause protective relay misoperation:

```python
# metrics/frequency.py
def cvar_95(errors: np.ndarray) -> float:
    """Mean of the worst 5% absolute errors (Conditional Value at Risk)."""
    threshold = np.percentile(np.abs(errors), 95)
    tail = np.abs(errors)[np.abs(errors) >= threshold]
    return float(tail.mean()) if len(tail) > 0 else float("inf")
```

### 11.5 Paired Statistical Design

When comparing method A and method B using the same MC seeds, observations are paired. The Wilcoxon signed-rank test is more powerful and statistically correct than an independent-sample test:

```python
from scipy.stats import wilcoxon
stat, pval = wilcoxon(rmse_per_seed_A, rmse_per_seed_B)
```

---

## 12. Open-Source Readiness Gaps

### 12.1 Critical — Blockers for Any Public Announcement

| Gap | Detail |
|---|---|
| `ofb/benchmarks/runner.py` returns fake zeros | `run_benchmark()` produces `TVE_mean: 0.0` for all methods. Silent correctness failure. |
| `tests/` is effectively empty | 3 files; 2 are empty. Real tests are in `src/openfreqbench and not discovered by root-level pytest. |
| Package name / import name mismatch | `pip install openfreqbench` → `import ofb` is wrong. |
| No CI/CD | Deleted in `q1-arch-final` branch. No automated quality gate. |
| Artifacts tracked in git | `v1/PMU/artifacts/` and `openfreqbench.egg-info/` in version control. |
| `README.md` describes non-working features | Features documented as working do not work as described. |

### 12.2 Important — Required Before Stable Release

| Gap | Detail |
|---|---|
| No `CONTRIBUTING.md` | Contributors cannot determine how to add an estimator or scenario. |
| `docs/` is all empty stubs | No content in any documentation page. |
| No stable public API declared | No clarity on what is public vs internal. |
| No schema versioning | Artifact format can change without detection. |
| `environment.yml` uses Python 3.11; `pyproject.toml` uses 3.10+ | Inconsistency; clarify minimum. |
| No smoke-test subset | No quick validation suite for contributors. |
| No `CITATION.cff` or BibTeX entry | Cannot be cited in academic work. |
| `CHANGELOG.md` is empty | No release history. |

### 12.3 Good to Have — Before v1.0

| Gap | Detail |
|---|---|
| No external plugin support via entry points | Community estimators require forking the core package. |
| No benchmark suite versioning policy | Results from different releases cannot be compared reliably. |
| No transfer benchmark mode | Generalization evaluation absent. |
| No Bayesian tuning option | Grid search is impractical for high-dimensional estimators. |
| OpenDSS scenarios not implemented | `s2_ieee13/` through `s5_kundur/` are stubs. |

### 12.4 Repo Hygiene Items

```bash
# Must be gitignored immediately
echo "v1/PMU/artifacts/"         >> .gitignore
echo "openfreqbench.egg-info/"   >> .gitignore
echo "artifacts/"                >> .gitignore

# Must be removed from git tracking
git rm --cached -r openfreqbench.egg-info/
git rm --cached -r v1/PMU/artifacts/
```

---

## 13. Documentation Files to Create

### 13.1 `CLAUDE.md` — AI-Assisted Development Rules

**Purpose:** Guardrails for AI-assisted development sessions. Prevents architectural regressions.

**Must contain:**
- Project purpose and scientific identity
- The three-world problem and the canonical direction
- Package decision: `src/openfreqbench/`, CLI `ofb`
- Dependency direction rules (which module may import what)
- Anti-patterns that must never be introduced:
  - Do not hardcode estimator or scenario lists in runners
  - Do not add plotting code to runners or metrics modules
  - Do not add framework imports inside estimator or scenario files
  - Do not modify artifact schemas without bumping `schema_version`
  - Do not add metric computation inside `_step()`
  - Do not bypass the `ArtifactIdentity` cache mechanism
- Benchmark philosophy: best-case-tuned mode is intentional; do not remove it
- What must never be broken: `BaseEstimator._step()` interface, `ScenarioBase.build()` interface, artifact identity scheme
- Testing invariant: every PR must pass `pytest -m smoke` and `ofb doctor`

**Why it matters:** AI-assisted sessions tend to introduce architecture drift. This file acts as a permanent constraint.

### 13.2 `ARCHITECTURE.md` — System Overview

**Purpose:** Technical reference for contributors and reviewers.

**Must contain:**
- System overview diagram (text-based, CLI → Core → Runners → Domain → I/O)
- Canonical package tree
- Domain model (entities, responsibilities, constraints)
- Dependency direction diagram
- The four-layer runner hierarchy with responsibilities and outputs
- Artifact store and cache key model
- Plugin system: auto-scan + entry points
- How a `benchmark.yaml` maps to execution
- Migration status table (canonical vs legacy vs deprecated)
- Where future features belong (Bayesian tuning → `tuning/bayesian.py`, GPU → `profiling/cuda.py`, etc.)

**Why it matters:** Without this document, contributors spend days reverse-engineering the architecture.

### 13.3 `BENCHMARK_SPEC.md` — Formal Benchmark Protocol

**Purpose:** Reproducibility and peer-review reference.

**Must contain:**
- Mode 1: Best-Case Tuned Per-Scenario (calibration seed, GSO protocol, held-out evaluation)
- Mode 2: Transfer Benchmark (train/eval split, no re-tuning policy)
- Mode 3: Sensitivity (perturbation protocol)
- Mode 4: Profiling (timing, memory, latency measurement protocols)
- Monte Carlo protocol: n_runs, seed strategy, warmup, minimum validity criteria
- Timing policy: `perf_counter_ns`, N warmup iterations, report p50/p95/p99
- Memory policy: RSS delta + tracemalloc peak after N samples
- Latency policy: structural = `latency_samples`, empirical = cross-correlation
- Metric definitions: RMSE_Hz, MAE_Hz, FE_Max_mHz, ROCOF_RMSE, IEEE_pass_rate, TRD, CVaR_95
- Causal alignment: how estimator latency is accounted for before error computation
- Artifact persistence rules per run level
- Schema versioning policy

**Why it matters:** A reviewer must be able to verify any result using only the spec and the code.

### 13.4 `CONTRIBUTING.md` — Contributor Guide

**Purpose:** On-ramp for external contributors.

**Must contain:**
- Prerequisites and environment setup (conda or pip)
- `git clone`, `pip install -e ".[dev]"`, `ofb doctor`
- How to add an estimator (step by step: `ofb new estimator`, implement, test, PR)
- How to add a scenario (step by step: `ofb new scenario`, implement, test, PR)
- How to add a metric
- Naming conventions: estimator files are snake_case; scenario IDs are `G{n}_E{n}_{Name}`
- Tests required per estimator (convergence, no NaN, determinism, reset)
- Tests required per scenario (shape, bounds, phase continuity)
- How to run a benchmark subset locally: `ofb run --config examples/quick_smoke.yaml`
- CI requirements: all checks must pass before merge

**Why it matters:** Without this, contributors cannot participate regardless of motivation.

### 13.5 `ROADMAP.md` — Technical Roadmap

**Purpose:** Transparency for users and contributors about priorities and direction.

**Must contain:**
- P0: Pre-launch blockers (fix runner stub, port domain, restore CI, gitignore artifacts)
- P1: Architectural upgrades (cache/resume, parallelism, YAML config, stats upgrades, transfer mode)
- P2: Advanced scientific improvements (N-way ANOVA, PCA, CVaR, Pareto, compliance benchmark)
- Release milestones (v0.2.0: working science pipeline; v0.3.0: full runner hierarchy; v1.0.0: public launch)
- Long-term items: GPU support, OpenDSS scenarios, community plugin registry

**Why it matters:** Establishes expectations and prevents scope creep in early contributions.

### 13.6 `RESULT_SCHEMA.md` — Artifact Schema Reference

**Purpose:** Specification for consumers of benchmark outputs and for schema versioning.

**Must contain:**
- Raw trace CSV schema: column names, types, units, `valid` flag definition
- Run metadata JSON schema: all required fields with types and examples
- Scenario-method summary JSON schema: metric fields, CI fields, profiling fields
- Suite summary schema
- `ArtifactIdentity` / cache key specification
- Schema version policy: what changes require a version bump
- Stale artifact detection: which fields make a cached result invalid
- How to re-analyze from saved artifacts without re-running simulation
- Naming conventions: snake_case, Hz units, `_us` suffix for microseconds

**Why it matters:** Downstream analysis tools, paper figures, and comparisons across releases all depend on schema stability.

### 13.7 `TESTING.md` — Testing Strategy

**Purpose:** Guide for contributors writing tests and for CI configuration.

**Must contain:**
- Test categories: `smoke`, `unit`, `integration`, `regression`, `slow`
- Smoke tests: import all estimators, run `ofb doctor`, run G1×ZC×3 seeds in < 60s
- Unit tests per estimator: convergence, no NaN, determinism, reset clears state
- Unit tests per scenario: shape consistency, `f_true` within nominal bounds, no inf/nan in `v`
- Metric tests: RMSE = 0 for perfect estimator, RMSE > 0 for noisy
- Runner integration tests: `TraceRunner` outputs valid CSV; metadata JSON is complete
- Scientific invariant tests: pure sine → RMSE < 0.01 Hz for baseline estimators
- Regression tests: lock RMSE of reference estimators on reference scenarios within tolerance
- CLI tests: `ofb run`, `ofb status`, `ofb new estimator` produce expected outputs
- Legacy parity tests: ported estimators match openfreqbench numerically on same inputs
- How to run categories: `pytest -m smoke`, `pytest -m unit`, `pytest -m slow`

---

## 14. Proposed Folder / Package Structure

```
open-freq-bench/                      # repo root
│
├── src/                              # ALL installable source code
│   └── openfreqbench/
│       ├── __init__.py
│       ├── _version.py
│       ├── cli/
│       ├── core/
│       ├── estimators/
│       ├── scenarios/
│       ├── metrics/
│       ├── profiling/
│       ├── tuning/
│       ├── runners/
│       ├── stats/
│       ├── reports/
│       ├── plotting/
│       ├── io/
│       └── compat/
│
├── tests/                            # pytest discovers only this tree
│   ├── conftest.py                   # shared fixtures
│   ├── markers.py                    # @pytest.mark.smoke, slow, integration
│   ├── smoke/
│   │   ├── test_cli_doctor.py
│   │   ├── test_imports.py
│   │   └── test_quick_run.py
│   ├── unit/
│   │   ├── estimators/               # one test file per estimator
│   │   ├── scenarios/                # one test file per scenario
│   │   ├── metrics/
│   │   └── profiling/
│   ├── integration/
│   │   ├── test_trace_runner.py
│   │   ├── test_scenario_method_runner.py
│   │   └── test_cli_run.py
│   └── regression/
│       └── test_reference_rmse.py    # golden RMSE values locked per estimator × scenario
│
├── examples/
│   ├── quick_smoke.yaml              # 1 scenario, 1 method, 5 MC runs
│   ├── baseline_g1.yaml             # G1 full suite, all methods
│   ├── full_benchmark.yaml          # all scenarios, all methods, 100 MC runs
│   ├── notebooks/
│   │   ├── 01_quick_start.ipynb
│   │   ├── 02_add_estimator.ipynb
│   │   └── 03_analyze_results.ipynb
│   └── scripts/
│       └── run_baseline.sh
│
├── docs/
│   ├── index.md
│   ├── install.md
│   ├── api/                          # auto-generated from docstrings
│   └── assets/
│
├── legacy/                           # archived, not packaged, not on sys.path
│   ├── README.md                     # "Archived. Do not import directly."
│   ├── openfreqbench/                     # src/openfreqbench moved here post-migration
│   ├── estimators_old/               # old PMU phasor contract estimators
│   └── scenarios_old/               # old tuple-returning scenario generators
│
├── artifacts/                        # GITIGNORED — generated benchmark outputs
│   └── .gitkeep
│
├── paper/                            # paper assets — separate from code
│   ├── paper.md
│   ├── paper.bib
│   └── figures/                      # final paper figures (committed)
│
├── docker/
│   └── Dockerfile
│
├── scripts/
│   ├── install.sh
│   ├── clean.sh
│   └── ci/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # lint + type check + test
│   │   ├── release.yml               # build + publish to PyPI
│   │   └── docs.yml                  # build + deploy docs
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
│
├── ARCHITECTURE.md
├── BENCHMARK_SPEC.md
├── CHANGELOG.md
├── CLAUDE.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── RESULT_SCHEMA.md
├── ROADMAP.md
├── TESTING.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .ruff.toml
└── mypy.ini
```

### Layout Notes

| Area | Rule |
|---|---|
| `src/` | All installable Python source. `pyproject.toml` configures `package-dir = {"" = "src"}`. |
| `tests/` | Root-level; `pytest` discovers from here. Never under `src/`. |
| `examples/` | Runnable YAML configs, notebooks, scripts. Not imported by `openfreqbench`. |
| `legacy/` | Not on `sys.path`. Not packaged. Contains archived openfreqbench and old estimators. |
| `artifacts/` | Gitignored. Generated benchmark outputs. Never committed. |
| `paper/` | Paper assets separate from source code. Final figures can be committed; raw data cannot. |
| `docs/` | Documentation source. Built by MkDocs or Sphinx during CI. |

**Gitignored (enforce in `.gitignore`):**
```
artifacts/
*.egg-info/
.venv*/
__pycache__/
.ofb/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.pyc
*.pyo
v1/PMU/artifacts/
```

---

## 15. Incremental Migration Plan

### 15.1 Guiding Principle

Migrate one vertical slice at a time. Each slice covers one estimator + one scenario + matching tests, end-to-end from raw signal to persisted artifact. Verify parity numerically before moving on. The openfreqbench test suite serves as the regression anchor throughout the migration.

### 15.2 Phase 0 — Immediate Hygiene (1–2 days)

**Goal:** Clean repo state, CI restored, artifacts gitignored.

- Add `v1/PMU/artifacts/`, `openfreqbench.egg-info/`, `artifacts/` to `.gitignore`
- `git rm --cached -r openfreqbench.egg-info/`
- Restore a minimal `.github/workflows/ci.yml` (lint + smoke test)
- Create `src/openfreqbench/` directory; update `pyproject.toml` package-dir
- Create `legacy/` directory with a `README.md`

**Expected outcome:** `git status` clean after install; CI badge green on a smoke-only test.

### 15.3 Phase 1 — Port Domain Core, First Vertical Slice (3–5 days)

**Goal:** One estimator + one scenario + one test passing end-to-end under new package.

**Actions (in order):**
1. Port `openfreqbench/estimators/base.py` → `src/openfreqbench/estimators/_base.py`
   - Remove `step()` self-timing
   - Remove `optimize()` method
   - Fix imports
2. Port `openfreqbench/scenarios/base.py` → `src/openfreqbench/scenarios/_base.py`
   - Fix imports only
3. Port `e1_zc.py` → `src/openfreqbench/estimators/zc.py`
4. Port `G1_E1_Pure_60Hz.py` → `src/openfreqbench/scenarios/g1/e1_pure_60hz.py`
5. Port matching tests to `tests/unit/estimators/test_zc.py` and `tests/unit/scenarios/test_g1_e1.py`
6. Add numerical parity test against openfreqbench original
7. Verify: `pytest tests/unit/estimators/test_zc.py tests/unit/scenarios/test_g1_e1.py -v`

**Expected outcome:** First vertical slice passes all tests with numerical parity to openfreqbench.

### 15.4 Phase 2 — Wire Real Runner (3–5 days)

**Goal:** `ofb run --config examples/quick_smoke.yaml` produces real RMSE values.

**Actions:**
- Extract `validate_estimator.optimize_estimator()` → `tuning/grid_search.py:TuningRunner`
- Extract `validate_estimator.run_mini_mc()` → `runners/scenario_method_runner.py`
- Extract `validate_estimator.run_statistical_analysis()` → `stats/monte_carlo.py`
- Extract `validate_estimator.save_artifacts()` → `io/artifact_store.py`
- Extract `validate_estimator.compare_methods()` → `reports/comparison.py`
- Replace `ofb/benchmarks/runner.py` stub with a call to the real `SuiteRunner`
- Wire `ofb run` CLI command to `SuiteRunner.run(config)`
- Add integration test: `test_cli_run.py`

**Expected outcome:** `ofb run --config examples/quick_smoke.yaml` produces non-zero RMSE in `artifacts/`.

### 15.5 Phase 3 — Port All Estimators and Scenarios (5–7 days)

**Goal:** All 10+ estimators and 18 scenarios accessible via CLI with passing tests.

- Port estimators e2 through e10 one at a time, with unit test each
- Port scenarios G2–G4 one at a time, with unit test each
- After each port: verify `ofb list estimators` and `ofb list scenarios` show the new entry
- Run full openfreqbench parity test for each ported estimator on matching scenario

**Expected outcome:** `ofb list estimators` shows 10+ methods; `ofb list scenarios` shows 18 scenarios; all parity tests pass.

### 15.6 Phase 4 — Cache, Resume, and Parallelism (3–5 days)

**Goal:** Incremental execution is reliable. Long runs can be interrupted and resumed.

- Implement `ArtifactIdentity` and `ArtifactStore` with identity-based paths
- Add skip-existing logic in `ScenarioMethodRunner`
- Implement `workers: N` in `SuiteRunner` via `ProcessPoolExecutor`
- Implement `ofb status` and `ofb resume` CLI commands
- Add integration tests for resume behavior

**Expected outcome:** Re-running a completed benchmark with `resume: true` logs "Skipping existing valid run" for all complete entries.

### 15.7 Phase 5 — Extract Profiling and Tuning (3–5 days)

**Goal:** Clean SRP. Timing and tuning are fully external to the estimator.

- Remove `perf_counter_ns` timing from `BaseEstimator.step()`
- Wire `TimingHarness` into `TraceRunner`
- Remove `BaseEstimator.optimize()` method
- Wire `TuningRunner.run()` into `ScenarioRunner`
- Wire `MemoryMeter` into `ScenarioMethodRunner`
- Verify timing results agree within 10% of previous behavior

**Expected outcome:** `BaseEstimator` has no knowledge of timing or tuning. Profiling data appears in `summary.json`.

### 15.8 Phase 6 — Statistics and Reporting Upgrades (3–5 days)

**Goal:** Statistically rigorous reports.

- Add `bonferroni_correction()` and `cohens_d()` to stats modules
- Add `cvar_95()` to metrics
- Add `ranking_stability()` at suite level
- Port `validate_estimator.py` plotting calls to `plotting/` module
- Verify `test_Metrics_bulletproof.py` parity plus new assertions on corrected values

**Expected outcome:** Report JSON includes `corrected_pvalue`, `cohens_d`, `cvar_95_hz` fields.

### 15.9 Phase 7 — Open-Source Launch Preparation (5–7 days)

**Goal:** Repository ready for public announcement.

- Write all documentation files (`CLAUDE.md`, `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `RESULT_SCHEMA.md`, `TESTING.md`)
- Move openfreqbench to `legacy/` (after Phase 3 parity is verified)
- Restore full CI/CD: lint, type check, smoke, unit, regression
- Rename all internal imports from `ofb.*` to `openfreqbench.*`
- Write `CITATION.cff`
- Release v0.2.0 on PyPI

**Expected outcome:** `pip install openfreqbench`; `ofb run --config examples/quick_smoke.yaml`; results in `artifacts/`; green CI badge.

### 15.10 Migration Risk Mitigation

**Numerical parity as anchor:**

```python
# tests/regression/test_openfreqbench_parity.py
def test_zc_numerical_parity():
    """Ported ZeroCrossing must match openfreqbench numerically on same input."""
    signal = np.sin(2 * np.pi * 60.0 * np.arange(1000) / 10000.0)
    new_result = ZeroCrossing({}).run(signal)
    old_result = ZeroCrossingEstimator().run(signal)
    np.testing.assert_allclose(new_result, old_result, rtol=1e-10)
```

**Parallel operation:** openfreqbench continues to work throughout migration. Researchers can use `src/openfreqbench while the new framework is being built. Archive to `legacy/` only after Phase 3 parity tests pass.

---

## 16. Prioritized Ticket Backlog (P0 / P1 / P2)

### P0 — Essential Before Any Public Launch

---

#### P0-01 — Fix runner stub: wire real Monte Carlo pipeline

| Field | Value |
|---|---|
| **Why it matters** | `ofb/benchmarks/runner.py` returns `TVE_mean: 0.0` for all runs. Silent correctness failure that discredits the project publicly. |
| **Affected modules** | `runners/`, `ofb/benchmarks/runner.py` |
| **Expected output** | `ofb run --config examples/quick_smoke.yaml` produces real RMSE values in `artifacts/` |
| **Acceptance criteria** | RMSE for G1×ZC is non-zero and matches openfreqbench output within 1% |
| **Migration risk** | Low — purely additive |
| **Blocks** | P0-02, P0-03, every other runner ticket |

---

#### P0-02 — Port all openfreqbench estimators to `src/openfreqbench/estimators/`

| Field | Value |
|---|---|
| **Why it matters** | 10 working estimators exist in openfreqbench but are not accessible via the CLI. |
| **Affected modules** | `estimators/`, `tests/unit/estimators/` |
| **Expected output** | `ofb list estimators` shows ZC, IZC, PM, IF_DPHI, ZC_MA, MWLS, RLS, WLS, AR, Prony, NR |
| **Acceptance criteria** | Each estimator passes its ported unit test from openfreqbench; numerical parity verified |
| **Migration risk** | Low — algorithm code unchanged; only imports and class names adjusted |
| **Blocks** | P0-01 |

---

#### P0-03 — Port all 18 openfreqbench scenarios to `src/openfreqbench/scenarios/`

| Field | Value |
|---|---|
| **Why it matters** | 18 working scenarios exist in openfreqbench but are not accessible via the CLI. |
| **Affected modules** | `scenarios/`, `tests/unit/scenarios/` |
| **Expected output** | `ofb list scenarios` shows G1–G4 groups |
| **Acceptance criteria** | Each scenario passes its ported unit test; `build(42).v.shape` and `build(42).f_true` are correct |
| **Migration risk** | Low |
| **Blocks** | P0-01 |

---

#### P0-04 — Port all openfreqbench tests to root-level `tests/`

| Field | Value |
|---|---|
| **Why it matters** | Real tests are invisible to `pytest` from the repo root. The project appears untested. |
| **Affected modules** | `tests/unit/` |
| **Expected output** | `pytest tests/unit/` runs 40+ tests and passes |
| **Acceptance criteria** | 100% pass rate on ported tests |
| **Migration risk** | None — test-only change |
| **Blocks** | None directly; enables regression anchors for all other tickets |

---

#### P0-05 — Fix package identity: rename `ofb/` to `src/openfreqbench/`

| Field | Value |
|---|---|
| **Why it matters** | `pip install openfreqbench` → `import ofb` is a confusing mismatch that blocks contributors. |
| **Affected modules** | All source files (imports), `pyproject.toml`, `README.md` |
| **Expected output** | `python -c "from openfreqbench import __version__; print(__version__)"` works |
| **Acceptance criteria** | No file imports `from ofb.` without DeprecationWarning; CI passes |
| **Migration risk** | Medium — requires touching all import statements |
| **Blocks** | Should be done early to avoid compounding renames |

---

#### P0-06 — Restore CI/CD pipeline

| Field | Value |
|---|---|
| **Why it matters** | Current branch deleted `.github/workflows/ci.yml`. No automated quality gate exists. |
| **Affected modules** | `.github/workflows/` |
| **Expected output** | CI runs on every PR: lint (ruff), type check (mypy), tests (`pytest -m smoke`) |
| **Acceptance criteria** | Green badge on main branch |
| **Migration risk** | None |
| **Blocks** | None |

---

#### P0-07 — Gitignore and untrack generated artifacts

| Field | Value |
|---|---|
| **Why it matters** | `v1/PMU/artifacts/` and `openfreqbench.egg-info/` tracked in git cause repo bloat and confusion. |
| **Affected modules** | `.gitignore` |
| **Expected output** | `git status` clean after `pip install -e .` |
| **Acceptance criteria** | No generated files appear in `git status` |
| **Migration risk** | None for source; artifacts may need to be distributed as release assets |
| **Blocks** | None |

---

#### P0-08 — Write minimal `CONTRIBUTING.md`

| Field | Value |
|---|---|
| **Why it matters** | Contributors cannot determine how to add an estimator or scenario. |
| **Affected modules** | `CONTRIBUTING.md` |
| **Expected output** | A contributor can follow the guide to add a new estimator in under two hours |
| **Acceptance criteria** | Document review passes; `ofb new estimator` workflow matches documentation |
| **Migration risk** | None |
| **Blocks** | None |

---

#### P0-09 — Implement YAML-driven benchmark configuration

| Field | Value |
|---|---|
| **Why it matters** | Benchmark matrix is hardcoded in Python. Researchers cannot customize without editing source. |
| **Affected modules** | `core/config.py`, `runners/suite_runner.py`, CLI `ofb run` |
| **Expected output** | `ofb run --config benchmark.yaml` works end-to-end with configurable scenario/method/MC params |
| **Acceptance criteria** | Changing `n_runs` in YAML changes MC repetitions without modifying any Python file |
| **Migration risk** | Low — new functionality |
| **Blocks** | P0-01 |

---

#### P0-10 — Add smoke test suite (`pytest -m smoke`, runs in < 60s)

| Field | Value |
|---|---|
| **Why it matters** | Contributors need quick validation. Full benchmark takes hours. |
| **Affected modules** | `tests/smoke/` |
| **Expected output** | `make smoke` completes in under 60 seconds |
| **Acceptance criteria** | Tests: import all estimators, run G1×ZC×3 seeds, check output shapes |
| **Migration risk** | None |
| **Blocks** | P0-06 (CI depends on this) |

---

### P1 — Strong Architectural Upgrades

---

#### P1-01 — Implement `ArtifactIdentity` cache key and resume logic

| Field | Value |
|---|---|
| **Why it matters** | No skip-existing mechanism; every run reruns from scratch; impractical for 100+ MC runs. |
| **Affected modules** | `core/run_identity.py`, `io/artifact_store.py`, `runners/trace_runner.py` |
| **Expected output** | Second run with same config skips existing valid artifacts |
| **Acceptance criteria** | `ofb run` logs "Skipping existing valid run: G1_E1 × ZC × seed=0" on second invocation |
| **Migration risk** | Low — additive |
| **Blocks** | P1-08 |

---

#### P1-02 — Extract tuning to standalone `TuningRunner` (SRP)

| Field | Value |
|---|---|
| **Why it matters** | `BaseEstimator.optimize()` violates SRP. Estimators should not own tuning logic. |
| **Affected modules** | `tuning/grid_search.py`, `estimators/_base.py` |
| **Expected output** | `TuningRunner(config).run(estimator_cls, v_cal, f_cal)` → `TuningResult` |
| **Acceptance criteria** | openfreqbench numerical parity test passes; `BaseEstimator` has no `optimize()` method |
| **Migration risk** | Medium — changes estimator base class |
| **Blocks** | P1-03 |

---

#### P1-03 — Extract timing to `TimingHarness` (SRP)

| Field | Value |
|---|---|
| **Why it matters** | Estimator self-timing is architecturally wrong and prevents fair external measurement. |
| **Affected modules** | `profiling/timing.py`, `estimators/_base.py` |
| **Expected output** | `TimingHarness.run_timed_trace(estimator, v_array)` → `TimingResult` |
| **Acceptance criteria** | Timing results agree within 10% of old approach; estimator base has no timing code |
| **Migration risk** | Medium |
| **Blocks** | None |

---

#### P1-04 — Implement `ProcessPoolExecutor` worker pool

| Field | Value |
|---|---|
| **Why it matters** | 18 × 12 × 100 = 21,600 sequential evaluations are impractical on a laptop. |
| **Affected modules** | `runners/worker_pool.py`, `runners/suite_runner.py` |
| **Expected output** | `workers: 4` in config dispatches 4 parallel `(scenario × method)` workers |
| **Acceptance criteria** | 4 workers produce a ≥ 3× speedup on a 4-core machine on the full G1 suite |
| **Migration risk** | Medium — requires artifact paths to be process-safe (no shared state) |
| **Blocks** | Depends on P1-01 (artifact store must be safe for parallel access) |

---

#### P1-05 — Add Bonferroni correction and effect sizes to statistical reports

| Field | Value |
|---|---|
| **Why it matters** | Pairwise p-values without correction inflate Type I error. Effect sizes required for scientific credibility. |
| **Affected modules** | `stats/hypothesis.py`, `stats/effects.py`, `reports/json_report.py` |
| **Expected output** | Report JSON includes `corrected_pvalue`, `cohens_d`, `reject_bonferroni` per method pair |
| **Acceptance criteria** | Numerically verified against manual computation on a synthetic dataset |
| **Migration risk** | Low — additive |
| **Blocks** | None |

---

#### P1-06 — Implement `ofb new estimator` and `ofb new scenario` scaffolding

| Field | Value |
|---|---|
| **Why it matters** | Contributors need a one-command flow to create correctly structured entities. |
| **Affected modules** | `cli/scaffold.py` |
| **Expected output** | `ofb new estimator --name my_ekf --family state_space` creates implementation file and test file |
| **Acceptance criteria** | Generated files pass ruff lint and mypy; test file imports and runs without error |
| **Migration risk** | None |
| **Blocks** | None |

---

#### P1-07 — Implement transfer benchmark mode

| Field | Value |
|---|---|
| **Why it matters** | Best-case tuned mode is complete. Transfer mode is needed for generalization evaluation. |
| **Affected modules** | `runners/`, `core/config.py`, `BENCHMARK_SPEC.md` |
| **Expected output** | `mode: transfer` in config tunes on G1 and evaluates on G2–G4 without re-tuning |
| **Acceptance criteria** | `ofb run --config transfer.yaml` produces `transfer_report.json` with G1-tuned params applied to G2–G4 |
| **Migration risk** | Low — additive; does not change existing Mode 1 behavior |
| **Blocks** | P0-09 |

---

#### P1-08 — Implement `ofb status` and `ofb resume`

| Field | Value |
|---|---|
| **Why it matters** | Long benchmark runs can be interrupted. Researchers need to inspect and resume progress. |
| **Affected modules** | `cli/commands/status.py`, `io/artifact_store.py` |
| **Expected output** | `ofb status --config benchmark.yaml` shows table of completed/missing/invalid runs |
| **Acceptance criteria** | `ofb resume --config benchmark.yaml` continues from last valid artifact after Ctrl+C interruption |
| **Migration risk** | Low |
| **Blocks** | P1-01 |

---

#### P1-09 — Write `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, `RESULT_SCHEMA.md`

| Field | Value |
|---|---|
| **Why it matters** | Without these, contributors cannot understand the system or verify reproducibility. |
| **Affected modules** | Root-level documentation |
| **Expected output** | Three complete Markdown documents |
| **Acceptance criteria** | A new contributor can understand where to add an estimator without asking questions; a reviewer can verify any result using the spec |
| **Migration risk** | None |
| **Blocks** | None |

---

#### P1-10 — Add CVaR-95 tail-risk metric

| Field | Value |
|---|---|
| **Why it matters** | RMSE underweights tail errors that cause protective relay misoperation. |
| **Affected modules** | `metrics/frequency.py`, result schemas |
| **Expected output** | `cvar_95_hz` field in all scenario-method summaries |
| **Acceptance criteria** | Numerically verified against manual computation |
| **Migration risk** | Low — additive |
| **Blocks** | None |

---

### P2 — Advanced Scientific and Benchmark Improvements

---

#### P2-01 — Add paired statistical testing (Wilcoxon signed-rank)

| Field | Value |
|---|---|
| **Why it matters** | Same-seed MC runs produce paired observations; paired tests are more powerful. |
| **Affected modules** | `stats/hypothesis.py` |
| **Expected output** | `paired_pvalue` field alongside `welch_pvalue` in comparison reports |
| **Acceptance criteria** | Verified on synthetic data where true effect is known |
| **Migration risk** | Low |
| **Blocks** | None |

---

#### P2-02 — Add ranking stability analysis (Spearman/Kendall across scenarios)

| Field | Value |
|---|---|
| **Why it matters** | Key research question: does "best on G1" generalize to "best on G4"? |
| **Affected modules** | `stats/ranking.py`, `reports/` |
| **Expected output** | Suite-level `ranking_stability_report.json` with per-metric Spearman correlations |
| **Acceptance criteria** | Matches manual computation on a small fixture dataset |
| **Migration risk** | Low |
| **Blocks** | None |

---

#### P2-03 — Add N-way ANOVA for method × scenario interaction

| Field | Value |
|---|---|
| **Why it matters** | Key scientific question: does method performance depend on scenario type? |
| **Affected modules** | `stats/hypothesis.py`, `reports/` |
| **Expected output** | `anova_report.json` with F-statistic, p-value, eta-squared per factor |
| **Acceptance criteria** | Matches `scipy.stats.f_oneway` result on same data |
| **Migration risk** | Low |
| **Blocks** | Requires full suite results (G1–G4 complete) |

---

#### P2-04 — Implement Pareto frontier figure (precision vs latency)

| Field | Value |
|---|---|
| **Why it matters** | The Pareto plot is the primary engineering visualization for method selection. |
| **Affected modules** | `plotting/pareto.py` |
| **Expected output** | `pareto_frontier.pdf` per benchmark suite identifying Pareto-optimal methods |
| **Acceptance criteria** | Pareto-optimal set is correctly identified relative to non-dominated sorting |
| **Migration risk** | None |
| **Blocks** | Requires profiling data (P1-03) and full RMSE results |

---

#### P2-05 — Add violin plots and KDE for error distributions

| Field | Value |
|---|---|
| **Why it matters** | RMSE hides distribution shape. Tail behavior and multi-modality are scientifically important. |
| **Affected modules** | `plotting/violin.py` |
| **Expected output** | Per-scenario violin plot with KDE overlay in suite report |
| **Acceptance criteria** | KDE integrates to 1.0 over the displayed range |
| **Migration risk** | None |
| **Blocks** | None |

---

#### P2-06 — Implement IEEE C37.118 compliance benchmark

| Field | Value |
|---|---|
| **Why it matters** | Regulatory compliance is a key claim for industrial relevance and is referenced in the research manifesto. |
| **Affected modules** | `metrics/compliance.py`, `scenarios/` |
| **Expected output** | Per-method `ieee_pass_rate` and `trd_ms` (Trip-Risk Duration) in compliance report |
| **Acceptance criteria** | Matches manual IEEE C37.118 envelope evaluation on reference waveforms |
| **Migration risk** | Low — additive |
| **Blocks** | Requires standard-specified scenario waveforms |

---

#### P2-07 — Implement external plugin support via entry points

| Field | Value |
|---|---|
| **Why it matters** | Community contributions should not require forking the core package. |
| **Affected modules** | `core/registry.py` |
| **Expected output** | A third-party pip-installable package can expose estimators via `[project.entry-points."openfreqbench.estimators"]` |
| **Acceptance criteria** | `pip install openfreqbench-my-ekf` → `ofb list estimators` shows `my_ekf` |
| **Migration risk** | Low — additive to registry |
| **Blocks** | P0-05 (registry must use canonical package name) |

---

#### P2-08 — Add Bayesian hyperparameter optimization option

| Field | Value |
|---|---|
| **Why it matters** | Grid search is exponential. Bayesian optimization is necessary for high-dimensional estimators (EKF, UKF, etc.). |
| **Affected modules** | `tuning/bayesian.py`, `TuningConfig` |
| **Expected output** | `strategy: bayesian` in `tuning:` config block uses Optuna or scikit-optimize |
| **Acceptance criteria** | Finds equivalent or better result than grid search in fewer evaluations on EKF |
| **Migration risk** | Low — additive; does not change grid search behavior |
| **Blocks** | P1-02 (TuningRunner must be extracted first) |

---

#### P2-09 — Port IEEE 13-bus OpenDSS scenarios

| Field | Value |
|---|---|
| **Why it matters** | Realistic distribution system validation beyond synthetic signals. Needed for G5 scenario group. |
| **Affected modules** | `scenarios/g5/` |
| **Expected output** | `G5_E1_Fault_SLG` and related scenarios produce physically valid waveforms via OpenDSS |
| **Acceptance criteria** | `G5_E1_Fault_SLG.build(seed=42)` produces waveform with correct fault dynamics |
| **Migration risk** | Medium — requires `dss-python` optional dependency and simulation validation |
| **Blocks** | None |

---

## 17. Conclusion

The scientific foundations of this project are sound and substantially complete. Eighteen scenario implementations, ten-plus estimator implementations, a full Monte Carlo runner, and a real statistical analysis pipeline already exist in `src/openfreqbench The test suite is comprehensive. Real benchmark artifacts with non-trivial results have been produced.

The primary gap is not scientific: it is engineering consolidation. The working scientific core in `openfreqbench` has not been promoted into the canonical `openfreqbench` package, the CLI is wired to stub code, and the test suite is invisible to standard tooling. These are correctable, incremental problems.

**The migration path does not require a reckless rewrite.** It requires:
1. Promoting `openfreqbench` domain logic into `src/openfreqbench/` while verifying numerical parity
2. Replacing the runner stub with the real orchestration logic
3. Wiring the existing CLI to the real runners
4. Establishing the cache/resume mechanism as a first-class concern
5. Restoring CI and writing the missing documentation

Each of these steps can be done independently without breaking the others. The openfreqbench codebase continues to function as a reference and regression anchor throughout the migration.

When the migration is complete, `openfreqbench` will be a serious, modular, contributor-friendly benchmark platform: a single `ofb run --config benchmark.yaml` command will produce statistically rigorous, reproducible, publication-grade comparisons across the full scenario and estimator matrix, with support for incremental execution, caching, parallelism, and extensibility.

---

*End of report. File version 1.0 — 2026-03-18.*
