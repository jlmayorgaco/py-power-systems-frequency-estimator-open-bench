# OpenFreqBench — Ticket Backlog and Roadmap

---

| Field | Value |
|---|---|
| **Status** | Active — pre-launch planning |
| **Source** | Derived from `openfreqbench_architecture_audit.md` |
| **Date** | 2026-03-18 |
| **Priority scheme** | P0 = must-have before public launch; P1 = strong architectural upgrade; P2 = advanced scientific improvement |

---

## Table of Contents

1. [Priority Legend](#1-priority-legend)
2. [P0 — Pre-Launch Blockers](#2-p0--pre-launch-blockers)
3. [P1 — Architectural Upgrades](#3-p1--architectural-upgrades)
4. [P2 — Advanced Scientific Improvements](#4-p2--advanced-scientific-improvements)
5. [Release Milestones](#5-release-milestones)
6. [Migration Phases Quick Reference](#6-migration-phases-quick-reference)
7. [Dependency Graph](#7-dependency-graph)
8. [Per-Module Work Summary](#8-per-module-work-summary)

---

## 1. Priority Legend

| Label | Meaning |
|---|---|
| **P0** | Blocking public launch. No release until all P0 tickets are closed. |
| **P1** | Required for architectural soundness. Must be complete by v0.3.0. |
| **P2** | Advanced features and scientific upgrades. Target v1.0.0. |
| **BLOCKS** | Listed tickets cannot start until this one is closed. |
| **DEPENDS ON** | This ticket requires another to be complete first. |

---

## 2. P0 — Pre-Launch Blockers

### P0-01 — Fix runner stub: wire real Monte Carlo pipeline

```
Status:   OPEN
Priority: P0
```

**Why it matters**
`ofb/benchmarks/runner.py` returns `TVE_mean: 0.0` for every run. The CLI silently produces
zeroed-out results with no error message. This is a correctness failure that would immediately
discredit the project.

**Root cause**
The real Monte Carlo pipeline (`run_mini_mc`, `run_statistical_analysis`, `save_artifacts`,
`compare_methods`) lives in `src/openfreqbench and has never
been wired to the CLI or the modern `ofb/` package.

**Affected modules**
- `ofb/benchmarks/runner.py` (replace stub)
- `src/openfreqbench/runners/suite_runner.py` (new)
- `src/openfreqbench/runners/scenario_method_runner.py` (new)

**Actions**
- [ ] Extract `validate_estimator.run_mini_mc()` → `runners/scenario_method_runner.py`
- [ ] Extract `validate_estimator.run_statistical_analysis()` → `stats/monte_carlo.py`
- [ ] Extract `validate_estimator.save_artifacts()` → `io/artifact_store.py`
- [ ] Extract `validate_estimator.compare_methods()` → `reports/comparison.py`
- [ ] Replace `ofb/benchmarks/runner.py` with a call to real `SuiteRunner`
- [ ] Wire `ofb run` CLI command to `SuiteRunner.run(config)`
- [ ] Add integration test: `tests/integration/test_cli_run.py`

**Acceptance criteria**
- `ofb run --config examples/quick_smoke.yaml` produces non-zero RMSE values
- RMSE for G1×ZC matches openfreqbench output within 1%
- `tests/integration/test_cli_run.py` passes

**Migration risk:** Low — purely additive
**Blocks:** P0-02, P0-03, P0-09, P1-01, P1-07

---

### P0-02 — Port all openfreqbench estimators to `src/openfreqbench/estimators/`

```
Status:   OPEN
Priority: P0
Depends on: P0-05 (package rename)
```

**Why it matters**
Ten working estimator implementations exist in `src/openfreqbench but are not
accessible from the CLI or the public package.

**Estimators to port**

| openfreqbench file | Target file | Class name |
|---|---|---|
| `e1_zc.py` | `estimators/zc.py` | `ZeroCrossing` |
| `e2_izc.py` | `estimators/izc.py` | `InterpolatedZeroCrossing` |
| `e3_pm.py` | `estimators/pm.py` | `PeriodMeasurement` |
| `e4_if_dphi.py` | `estimators/if_dphi.py` | `InstantaneousFrequencyPhaseIncrement` |
| `e5_zc_ma.py` | `estimators/zc_ma.py` | `ZeroCrossingMovingAverage` |
| `e6_mwls.py` | `estimators/mwls.py` | `MovingWindowLeastSquares` |
| `e7_rls_basic.py` | `estimators/rls.py` | `RecursiveLeastSquares` |
| `e7_wls.py` | `estimators/wls.py` | `WindowedLeastSquares` |
| `e8_ar.py` | `estimators/ar.py` | `Autoregressive` |
| `e9_prony.py` | `estimators/prony.py` | `Prony` |
| `e10_nr.py` | `estimators/nr.py` | `NewtonRaphson` |

**Actions per estimator**
- [ ] Copy file, rename class, fix imports
- [ ] Add numerical parity test in `tests/regression/test_openfreqbench_parity.py`
- [ ] Verify `ofb list estimators` shows new entry

**Acceptance criteria**
- All 10+ estimators visible in `ofb list estimators`
- All unit tests pass
- All parity tests pass

**Migration risk:** Low
**Blocks:** P0-01

---

### P0-03 — Port all 18 openfreqbench scenarios to `src/openfreqbench/scenarios/`

```
Status:   OPEN
Priority: P0
Depends on: P0-05
```

**Why it matters**
18 working scenario implementations exist in `src/openfreqbench but are not
accessible from the CLI.

**Scenarios to port**

| Group | openfreqbench file | Target path |
|---|---|---|
| G1 | `G1_E1_Pure_60Hz.py` | `scenarios/g1/e1_pure_60hz.py` |
| G1 | `G1_E2_Gaussian_Noise_1pct.py` | `scenarios/g1/e2_gaussian_noise_1pct.py` |
| G1 | `G1_E3_Gaussian_Noise_5pct.py` | `scenarios/g1/e3_gaussian_noise_5pct.py` |
| G2 | `G2_E4_Voltage_Mag_Step_1pct.py` | `scenarios/g2/e4_voltage_step_1pct.py` |
| G2 | `G2_E5_Voltage_Mag_Step_10pct.py` | `scenarios/g2/e5_voltage_step_10pct.py` |
| G2 | `G2_E6_Freq_Step_60_to_59p5.py` | `scenarios/g2/e6_freq_step_0p5hz.py` |
| G2 | `G2_E7_Freq_Step_60_to_55.py` | `scenarios/g2/e7_freq_step_5hz.py` |
| G2 | `G2_E8_Fast_Ramp_plus5Hzs.py` | `scenarios/g2/e8_fast_ramp_5hzs.py` |
| G2 | `G2_E9_Slow_Ramp_minus0p5Hzs.py` | `scenarios/g2/e9_slow_ramp_0p5hzs.py` |
| G3 | `G3_E10_AM_Modulation.py` | `scenarios/g3/e10_am_modulation.py` |
| G3 | `G3_E11_FM_Modulation.py` | `scenarios/g3/e11_fm_modulation.py` |
| G3 | `G3_E12_Phase_Jump.py` | `scenarios/g3/e12_phase_jump.py` |
| G3 | `G3_E13_Impulsive_Outliers.py` | `scenarios/g3/e13_impulsive_outliers.py` |
| G3 | `G3_E14_Noise_Harmonics.py` | `scenarios/g3/e14_noise_harmonics.py` |
| G3 | `G3_E15_Noise_Interharmonics.py` | `scenarios/g3/e15_noise_interharmonics.py` |
| G4 | `G4_E16_Composite_Islanding.py` | `scenarios/g4/e16_composite_islanding.py` |
| G4 | `G4_E17_Multi_Event_Profile.py` | `scenarios/g4/e17_multi_event_profile.py` |
| G4 | `G4_E18_Chamorro_Event.py` | `scenarios/g4/e18_chamorro_event.py` |

**Acceptance criteria**
- All 18 scenarios visible in `ofb list scenarios`
- Each scenario unit test passes
- `build(seed=42).v.shape` and `build(seed=42).f_true` are correct

**Migration risk:** Low
**Blocks:** P0-01

---

### P0-04 — Port all openfreqbench tests to root-level `tests/`

```
Status:   OPEN
Priority: P0
```

**Why it matters**
13 estimator tests and 18 scenario tests and metrics tests in `src/openfreqbench are
invisible to `pytest` at the repo root. The project appears untested to any contributor or CI.

**Tests to port**

| openfreqbench test | Target |
|---|---|
| `test_e1_zc.py` through `test_e10_nr.py` | `tests/unit/estimators/` |
| `test_Scenario_G1_E1_*.py` through `test_Scenario_G4_E18_*.py` | `tests/unit/scenarios/` |
| `test_Metrics_bulletproof.py` | `tests/unit/metrics/test_metrics.py` |

**Actions**
- [ ] Copy each test file, update imports to use `openfreqbench.*`
- [ ] Add `@pytest.mark.unit` to all
- [ ] Verify: `pytest tests/unit/ -v` runs and passes

**Acceptance criteria**
- `pytest tests/unit/` discovers 40+ tests and all pass

**Migration risk:** None
**Blocks:** None (enables regression anchors for all other tickets)

---

### P0-05 — Fix package identity: rename `ofb/` to `src/openfreqbench/`

```
Status:   OPEN
Priority: P0
```

**Why it matters**
`pip install openfreqbench` then `import ofb` is a confusing mismatch that will block
contributors immediately.

**Actions**
- [ ] Create `src/openfreqbench/` directory structure
- [ ] Move all files from `ofb/` to `src/openfreqbench/`
- [ ] Update `pyproject.toml`: `package-dir = {"" = "src"}`, `find.include = ["openfreqbench*"]`
- [ ] Replace all `from ofb.` imports with `from openfreqbench.`
- [ ] Add `compat/openfreqbench.py` shim with `DeprecationWarning`
- [ ] Update `README.md`, `CONTRIBUTING.md` to use new import path
- [ ] Verify: `python -c "from openfreqbench import __version__; print(__version__)"`

**Acceptance criteria**
- No file imports `from ofb.` without triggering DeprecationWarning
- CI passes after rename

**Migration risk:** Medium — all import statements must be updated
**Blocks:** Should be done early to avoid compounding renames

---

### P0-06 — Restore CI/CD pipeline

```
Status:   OPEN
Priority: P0
```

**Why it matters**
The branch `q1-arch-final` deleted `.github/workflows/ci.yml` and all related automation.
No automated quality gate exists.

**Minimum CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install ruff mypy
      - run: ruff check src/ tests/
      - run: mypy src/openfreqbench --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: pytest -m smoke -q
```

**Acceptance criteria**
- Green CI badge on main branch
- PRs blocked if lint or smoke tests fail

**Migration risk:** None
**Blocks:** None (P0-10 depends on this)

---

### P0-07 — Gitignore and untrack generated artifacts

```
Status:   OPEN
Priority: P0
```

**Why it matters**
`v1/PMU/artifacts/mini_mc_compare/` and `openfreqbench.egg-info/` are tracked in git.
This causes hundreds of lines of diff noise and confuses the history.

**Actions**
- [ ] Add to `.gitignore`:
  ```
  v1/PMU/artifacts/
  openfreqbench.egg-info/
  artifacts/
  *.egg-info/
  ```
- [ ] Run: `git rm --cached -r openfreqbench.egg-info/`
- [ ] Run: `git rm --cached -r v1/PMU/artifacts/`
- [ ] Consider distributing artifacts as GitHub Release assets for reproducibility reference

**Acceptance criteria**
- `git status` is clean after `pip install -e .`
- `artifacts/` directory exists locally but is gitignored

**Migration risk:** None for source; artifacts should be preserved externally

---

### P0-08 — Write minimal `CONTRIBUTING.md`

```
Status:   OPEN
Priority: P0
```

**Why it matters**
A contributor visiting the repository currently has no guidance on how to add an estimator,
run tests, or configure a development environment.

**Minimum required sections**
- [ ] Prerequisites (Python 3.11, conda or pip)
- [ ] Development install: `pip install -e ".[dev]"`
- [ ] Run smoke tests: `pytest -m smoke`
- [ ] How to add an estimator (step-by-step with `ofb new estimator`)
- [ ] How to add a scenario (step-by-step)
- [ ] Naming conventions
- [ ] Required tests per estimator and per scenario
- [ ] How to run a local benchmark subset: `ofb run --config examples/quick_smoke.yaml`
- [ ] PR checklist

**Acceptance criteria**
- A contributor can follow the guide to add a working estimator in under two hours without external help

---

### P0-09 — Implement YAML-driven benchmark configuration

```
Status:   OPEN
Priority: P0
Depends on: P0-01
```

**Why it matters**
The benchmark matrix (which scenarios, which methods, how many MC runs, what targets) is
hardcoded in Python scripts. External researchers cannot configure a run without editing source code.

**Target interface**
```bash
ofb run --config benchmark.yaml
```

**Config fields to support**
```yaml
benchmark:      {name, policy_version, schema_version, mode}
monte_carlo:    {n_runs, seeds_strategy, warmup_samples}
tuning:         {metric, strategy, max_seconds, stride, early_stop_rmse, patience}
profiling:      {measure_time, measure_memory, n_warmup_iters}
scenarios:      [{id, params}]
methods:        [{id, params}]
targets:        {RMSE_HZ_TARGET, IEEE_PASS_RATE_TARGET}
workers:        int
resume:         bool
output_dir:     str
```

**Actions**
- [ ] Extend `core/config.py` Pydantic models to cover all config fields
- [ ] Implement YAML loader with validation: `core/config.py:load_config(path)`
- [ ] Wire `ofb run --config PATH` to load config and pass to `SuiteRunner`
- [ ] Write `examples/quick_smoke.yaml` (1 scenario, 1 method, 5 runs)
- [ ] Write `examples/baseline_g1.yaml` (G1 full suite)
- [ ] Add CLI test: `test_cli_run.py::test_yaml_config_loads`

**Acceptance criteria**
- Changing `n_runs` in YAML changes MC repetitions without modifying Python
- Invalid YAML produces a clear validation error before any benchmark runs

---

### P0-10 — Add smoke test suite

```
Status:   OPEN
Priority: P0
Depends on: P0-06
```

**Why it matters**
There is no quick validation path for contributors. Full benchmarks take hours.

**Target**
```bash
pytest -m smoke    # completes in < 60 seconds
make smoke         # alias
```

**Smoke test contents**
- [ ] `test_imports.py`: import all estimators, import all scenarios — no ImportError
- [ ] `test_cli_doctor.py`: `ofb doctor` exits with code 0
- [ ] `test_quick_run.py`: run G1_E1_Pure_60Hz × ZeroCrossing × 3 seeds; check output shape and RMSE > 0

**Acceptance criteria**
- `make smoke` completes in under 60 seconds on a standard laptop
- CI smoke job passes

---

## 3. P1 — Architectural Upgrades

### P1-01 — Implement `ArtifactIdentity` cache key and resume logic

```
Status:   OPEN
Priority: P1
Depends on: P0-01
```

**Why it matters**
Every run currently starts from scratch. The runner uses `ts_now()` as part of the artifact
path, so re-runs create new directories instead of skipping existing results. This blocks the
incremental `scenario × method` workflow that is central to the research process.

**Design**
```python
@dataclasses.dataclass(frozen=True)
class ArtifactIdentity:
    scenario_id: str
    scenario_config_hash: str   # SHA-256 of scenario params JSON
    method_id: str
    method_config_hash: str     # SHA-256 of estimator params JSON
    seed: int
    policy_version: str
    schema_version: str

    @property
    def cache_key(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

**Actions**
- [ ] Implement `ArtifactIdentity` in `core/run_identity.py`
- [ ] Implement `ArtifactStore.exists_valid(identity) → bool`
- [ ] Implement `ArtifactStore.load_trace(identity) → TraceResult`
- [ ] Implement `ArtifactStore.save_trace(identity, result)` with identity-based path
- [ ] Add skip-existing logic in `ScenarioMethodRunner`
- [ ] Add `ofb status` command (basic table of complete/missing runs)

**Acceptance criteria**
- Second run with same config logs: `Skipping existing valid run: G1_E1 × ZC × seed=0`
- `ofb status --config benchmark.yaml` prints completion table

**Blocks:** P1-08

---

### P1-02 — Extract tuning to standalone `TuningRunner`

```
Status:   OPEN
Priority: P1
```

**Why it matters**
`BaseEstimator.optimize()` violates the Single Responsibility Principle. An estimator should not
own its own tuning logic. Tuning is a framework concern.

**Target design**
```python
# tuning/grid_search.py
class TuningRunner:
    def __init__(self, config: TuningConfig): ...
    def run(
        self,
        estimator_cls: type[BaseEstimator],
        v_cal: np.ndarray,
        f_cal: np.ndarray,
        fs_hz: float,
    ) -> TuningResult:
        """Grid search over TuningParam ranges. Returns best params and score."""
```

**Actions**
- [ ] Create `tuning/grid_search.py` with `TuningRunner`
- [ ] Port GSO logic from `BaseEstimator.optimize()` into `TuningRunner.run()`
- [ ] Remove `optimize()` from `BaseEstimator` in `estimators/_base.py`
- [ ] Wire `TuningRunner` into `ScenarioRunner` (step 1 of each method loop)
- [ ] Add openfreqbench parity test: GSO produces same best params on same input

**Acceptance criteria**
- `BaseEstimator` has no `optimize()` method
- Parity test passes

**Blocks:** P2-08 (Bayesian tuning)

---

### P1-03 — Extract timing to `TimingHarness`

```
Status:   OPEN
Priority: P1
```

**Why it matters**
Estimator self-timing is architecturally wrong. The estimator should have no awareness of how
it is being measured. Timing contamination within `step()` also slightly affects the measurement.

**Target design**
```python
# profiling/timing.py
class TimingHarness:
    def run_timed_trace(
        self, estimator: BaseEstimator, v_array: np.ndarray, n_warmup: int = 3
    ) -> TimingResult: ...
```

**Actions**
- [ ] Create `profiling/timing.py` with `TimingHarness` and `TimingResult`
- [ ] Remove `perf_counter_ns` timing from `BaseEstimator.step()`
- [ ] Wire `TimingHarness` into `TraceRunner`
- [ ] Verify timing results agree within 10% of previous approach

**Acceptance criteria**
- `BaseEstimator.step()` contains no timing code
- `TimingResult` appears in `trace.csv` and `summary.json`

---

### P1-04 — Implement `ProcessPoolExecutor` worker pool

```
Status:   OPEN
Priority: P1
Depends on: P1-01 (artifact store must be process-safe)
```

**Why it matters**
18 × 12 × 100 = 21,600 sequential evaluations are impractical. A 4-worker pool can reduce
wall-clock time by ~3–3.5×.

**Worker granularity:** `(scenario × method)` — each worker has its own artifact paths; no shared state.

**Actions**
- [ ] Implement `runners/worker_pool.py` with `ProcessPoolExecutor` dispatch
- [ ] Update `SuiteRunner` to accept `workers: int` from config
- [ ] Test that 4-worker run produces identical results to 1-worker run
- [ ] Handle worker exceptions without killing the entire suite

**Acceptance criteria**
- `workers: 4` in config provides ≥ 3× speedup on G1 suite (4-core machine)
- Failed `(scenario × method)` does not abort other workers

---

### P1-05 — Add Bonferroni correction and effect sizes to statistical reports

```
Status:   OPEN
Priority: P1
```

**Why it matters**
Pairwise p-values without multiple-comparison correction inflate Type I error when comparing
many methods. Effect sizes are required for scientific credibility in peer review.

**Actions**
- [ ] Implement `stats/hypothesis.py::bonferroni_correction(pvalues, alpha)` → dict
- [ ] Implement `stats/effects.py::cohens_d(a, b)` → float
- [ ] Wire both into `reports/comparison.py::compare_methods()`
- [ ] Add `corrected_pvalue`, `cohens_d`, `reject_bonferroni` to scenario comparison JSON
- [ ] Verify numerically against manual computation

**Acceptance criteria**
- Scenario comparison JSON includes `corrected_pvalue` and `cohens_d` per method pair
- Values match manual computation on a fixture dataset

---

### P1-06 — Implement `ofb new estimator` and `ofb new scenario` scaffolding

```
Status:   OPEN
Priority: P1
```

**Why it matters**
Scaffolding reduces contributor friction from hours to minutes and ensures new entities
follow the correct structure from the start.

**Commands**
```bash
ofb new estimator --name my_ekf --family state_space
ofb new scenario --id G5_E19_My_Event --group G5
ofb new metric --name my_cvar
```

**Generated files per estimator**
- `src/openfreqbench/estimators/my_ekf.py` — populated template
- `tests/unit/estimators/test_my_ekf.py` — populated test template

**Actions**
- [ ] Create Jinja2 or string templates for estimator, scenario, and metric
- [ ] Implement `cli/scaffold.py::scaffold_estimator(name, family)`
- [ ] Implement `cli/scaffold.py::scaffold_scenario(id, group)`
- [ ] Verify generated files pass `ruff check` and `mypy`
- [ ] Add CLI test: `ofb new estimator` creates expected files

**Acceptance criteria**
- Generated implementation file is syntactically correct and importable
- Generated test file runs without error after the contributor fills in `_step()`

---

### P1-07 — Implement transfer benchmark mode

```
Status:   OPEN
Priority: P1
Depends on: P0-09 (YAML config)
```

**Why it matters**
The best-case-tuned mode (Mode 1) is complete. Transfer mode (Mode 2) answers the distinct
question: does a method tuned on one scenario family generalize to another?

**Config**
```yaml
benchmark:
  mode: "transfer"

transfer:
  train_scenario_ids: ["G1_E1_Pure_60Hz", "G1_E2_Gaussian_Noise_1pct"]
  eval_scenario_ids:  ["G2_E7_Freq_Step_60_to_55", "G3_E12_Phase_Jump"]
```

**Behavior**
- Tune each method once on the train scenarios
- Evaluate on eval scenarios with fixed params (no re-tuning)
- Save separate `transfer_report.json` alongside best-case results

**Actions**
- [ ] Add `TransferConfig` to `core/config.py`
- [ ] Implement transfer mode in `ScenarioRunner`
- [ ] Add `transfer_report.json` schema
- [ ] Add CLI test: transfer run with 2 train + 2 eval scenarios

**Acceptance criteria**
- `ofb run --config transfer.yaml` produces `transfer_report.json`
- Same method, same train scenarios → identical tuned params in both Mode 1 and transfer runs

---

### P1-08 — Implement `ofb status` and `ofb resume`

```
Status:   OPEN
Priority: P1
Depends on: P1-01 (ArtifactIdentity)
```

**Why it matters**
Multi-hour benchmark runs can be interrupted. Researchers must be able to see progress and
resume without restarting from scratch.

**`ofb status` output example**
```
Benchmark: baseline_freq_v1
┌──────────────────────────┬─────────────────────┬──────────┬───────────┐
│ Scenario                 │ Method              │ Status   │ Seeds     │
├──────────────────────────┼─────────────────────┼──────────┼───────────┤
│ G1_E1_Pure_60Hz          │ ZeroCrossing        │ COMPLETE │ 100/100   │
│ G1_E1_Pure_60Hz          │ EKF                 │ PARTIAL  │ 47/100    │
│ G2_E7_Freq_Step_60_to_55 │ ZeroCrossing        │ MISSING  │ 0/100     │
└──────────────────────────┴─────────────────────┴──────────┴───────────┘
```

**Actions**
- [ ] Implement `cli/commands/status.py::status_command(config_path)`
- [ ] Implement `cli/commands/run.py::resume_command(config_path)` (runs only missing/partial)
- [ ] Add tests for status output format

**Acceptance criteria**
- `ofb status --config benchmark.yaml` prints correct completion table
- `ofb resume --config benchmark.yaml` after Ctrl+C continues from last valid artifact

---

### P1-09 — Write `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, `RESULT_SCHEMA.md`

```
Status:   OPEN
Priority: P1
```

**Why it matters**
Without these documents, a contributor cannot understand the architecture and a reviewer
cannot verify reproducibility.

**Required sections per file**

`ARCHITECTURE.md`:
- [ ] System overview diagram
- [ ] Canonical package tree
- [ ] Domain model table
- [ ] Dependency direction rules
- [ ] Four-layer runner hierarchy
- [ ] Artifact store and cache key model
- [ ] Plugin system design
- [ ] Migration status

`BENCHMARK_SPEC.md`:
- [ ] All benchmark modes with formal definitions
- [ ] Monte Carlo protocol
- [ ] Timing and memory measurement policies
- [ ] Metric definitions (RMSE_Hz, MAE_Hz, FE_Max, ROCOF, CVaR_95, TRD)
- [ ] Causal alignment policy
- [ ] Artifact persistence rules

`RESULT_SCHEMA.md`:
- [ ] Raw trace CSV schema
- [ ] Run metadata JSON schema
- [ ] Scenario-method summary JSON schema
- [ ] `ArtifactIdentity` / cache key spec
- [ ] Schema versioning policy

**Acceptance criteria**
- New contributor can understand where to add an estimator without asking
- Reviewer can verify any result using only the spec and the code

---

### P1-10 — Add CVaR-95 tail-risk metric

```
Status:   OPEN
Priority: P1
```

**Why it matters**
RMSE underweights the catastrophic tail errors that cause protective relay misoperation.
CVaR-95 (mean of worst 5% errors) directly quantifies this risk.

**Implementation**
```python
# metrics/frequency.py
def cvar_95(errors: np.ndarray) -> float:
    """Mean of the worst 5% absolute errors."""
    threshold = np.percentile(np.abs(errors), 95)
    tail = np.abs(errors)[np.abs(errors) >= threshold]
    return float(tail.mean()) if len(tail) > 0 else float("inf")
```

**Actions**
- [ ] Implement `cvar_95` in `metrics/frequency.py`
- [ ] Wire into `aggregate_monte_carlo()`
- [ ] Add `cvar_95_hz` to scenario-method summary JSON
- [ ] Verify numerically

**Acceptance criteria**
- `cvar_95_hz` present in all scenario-method summary files
- Value verified against manual computation on a fixture

---

## 4. P2 — Advanced Scientific Improvements

### P2-01 — Paired statistical testing (Wilcoxon signed-rank)

```
Status:   OPEN
Priority: P2
```

**Why it matters**
When A and B are tested with the same MC seeds, the RMSE values per seed are paired
observations. The Wilcoxon signed-rank test (or paired t-test) is more powerful than
the current unpaired Welch test.

**Actions**
- [ ] Implement `stats/hypothesis.py::wilcoxon_signed_rank(a, b)` → dict with `stat`, `pvalue`
- [ ] Wire into `compare_methods()` when paired seeds are detected
- [ ] Add `paired_pvalue` field to comparison JSON alongside `welch_pvalue`

**Acceptance criteria**
- `paired_pvalue` appears in comparison JSON when seed lists overlap
- Result verified on synthetic dataset with known true effect

---

### P2-02 — Ranking stability analysis (Spearman/Kendall across scenarios)

```
Status:   OPEN
Priority: P2
```

**Why it matters**
A key research question: does "best on G1" imply "best on G4"? If rankings are unstable,
no universal winner exists — a major and publishable finding.

**Actions**
- [ ] Implement `stats/ranking.py::ranking_stability(rankings_per_scenario)` → dict
- [ ] Report Spearman and Kendall correlation, mean, std across scenario pairs
- [ ] Add `ranking_stability_report.json` to suite-level artifacts
- [ ] Add discussion-prompt field: `"universal_winner": bool`

**Acceptance criteria**
- Spearman correlation values match manual computation on fixture
- Report is present in suite artifacts after a complete run

---

### P2-03 — N-way ANOVA for method × scenario interaction

```
Status:   OPEN
Priority: P2
Depends on: Full suite results (G1–G4 complete)
```

**Why it matters**
N-way ANOVA with interaction term answers: does method performance depend on scenario type?
A significant interaction term means methods are specialized, not universal — a core scientific claim.

**Actions**
- [ ] Implement `stats/hypothesis.py::nway_anova(data, factors)` → dict
- [ ] Wire into suite-level report generation
- [ ] Add `anova_report.json` with F-statistic, p-value, eta-squared per factor

**Acceptance criteria**
- Matches `scipy.stats.f_oneway` result on same data
- Report appears in suite artifacts

---

### P2-04 — Pareto frontier figure (precision vs latency)

```
Status:   OPEN
Priority: P2
Depends on: P1-03 (profiling data), full RMSE results
```

**Why it matters**
The Pareto plot is the definitive engineering visualization for method selection. It identifies
which methods offer the best precision-latency trade-off.

**Actions**
- [ ] Implement `plotting/pareto.py::plot_pareto_frontier(methods, rmse_dict, latency_dict)`
- [ ] Highlight Pareto-optimal methods
- [ ] Save `pareto_frontier.pdf` to suite artifacts

**Acceptance criteria**
- Pareto-optimal set correctly identified using standard non-dominated sorting
- Figure is publication quality (vector PDF, labeled axes, legend)

---

### P2-05 — Violin plots and KDE for error distributions

```
Status:   OPEN
Priority: P2
```

**Why it matters**
RMSE hides distribution shape. Tail behavior, multi-modality, and outlier presence are
scientifically important and directly relevant to the paper's CVaR and tail-risk claims.

**Actions**
- [ ] Implement `plotting/violin.py::plot_violin_per_scenario(summaries)`
- [ ] Overlay KDE estimate on each violin
- [ ] Save one figure per scenario group

**Acceptance criteria**
- KDE integrates to 1.0 over displayed range
- Figures use consistent color scheme across estimator families

---

### P2-06 — IEEE C37.118 compliance benchmark

```
Status:   OPEN
Priority: P2
```

**Why it matters**
The research manifesto proposes a new IEEE compliance standard for IBR-dominated grids.
The existing framework needs a compliance benchmark mode to support this claim.

**Metrics to add**
- `ieee_pass_rate`: fraction of samples within the C37.118 frequency error envelope
- `trd_ms`: Trip-Risk Duration — total time where error exceeds protective relay threshold

**Actions**
- [ ] Implement `metrics/compliance.py::ieee_pass_rate(f_est, f_true, fs_hz)` → float
- [ ] Implement `metrics/compliance.py::trip_risk_duration(f_est, f_true, fs_hz, threshold_hz)` → float
- [ ] Add standard-specified scenario waveforms (frequency step, ramp, modulation per C37.118)
- [ ] Add `compliance_report.json` to scenario-level artifacts

**Acceptance criteria**
- Values match manual IEEE C37.118 envelope evaluation on reference waveforms
- Compliance report present in artifacts for all methods on standard scenarios

---

### P2-07 — External plugin support via entry points

```
Status:   OPEN
Priority: P2
Depends on: P0-05 (canonical package name)
```

**Why it matters**
Community contributions should not require forking the core package. A researcher should be
able to publish `pip install openfreqbench-my-ekf` and have it appear in `ofb list estimators`.

**Implementation**
```python
# core/registry.py
import importlib.metadata
for ep in importlib.metadata.entry_points(group="openfreqbench.estimators"):
    cls = ep.load()
    register("estimator", ep.name, cls=cls)
```

**Actions**
- [ ] Add entry point scanning to `core/registry.py`
- [ ] Document the `[project.entry-points]` format in `CONTRIBUTING.md`
- [ ] Add test with a mock entry point

**Acceptance criteria**
- A mock third-party package exposing an entry point causes the estimator to appear in `ofb list estimators`

---

### P2-08 — Bayesian hyperparameter optimization

```
Status:   OPEN
Priority: P2
Depends on: P1-02 (TuningRunner extracted)
```

**Why it matters**
Grid search is exponential. For high-dimensional estimators (EKF, UKF with 5+ hyperparameters),
Bayesian optimization finds good configurations in far fewer evaluations.

**Config**
```yaml
tuning:
  strategy: "bayesian"   # or: "grid" (default)
  n_trials: 50
  timeout_s: 120
```

**Actions**
- [ ] Implement `tuning/bayesian.py::BayesianTuningRunner` using Optuna or scikit-optimize
- [ ] Wire to `TuningConfig::strategy`
- [ ] Verify Bayesian finds equivalent or better result than grid search on EKF in fewer trials

---

### P2-09 — Port IEEE 13-bus OpenDSS scenarios (G5 group)

```
Status:   OPEN
Priority: P2
```

**Why it matters**
Realistic distribution system scenarios beyond synthetic signals. Required for T+D
co-simulation claims in the research manifesto.

**Scenarios to implement**
- `G5_E1_Fault_SLG_Bus671`
- `G5_E2_Motor_Start`
- `G5_E3_PV_Ramp`
- `G5_E4_RegTap_Step`

**Actions**
- [ ] Implement `scenarios/g5/` using `dss-python` (optional dependency)
- [ ] Validate waveform physics against OpenDSS reference results
- [ ] Add optional-dependency guard: skip these tests if `dss-python` not installed

**Acceptance criteria**
- `G5_E1_Fault_SLG.build(seed=42)` produces physically valid fault waveform
- Tests marked `@pytest.mark.opendss` and skipped when `dss-python` is absent

---

## 5. Release Milestones

### v0.2.0 — Working Science Pipeline

**Target:** All P0 tickets closed.

**Definition of done:**
- [ ] P0-01 through P0-10 closed
- [ ] `ofb run --config examples/quick_smoke.yaml` produces real results
- [ ] `pytest -m smoke` passes in < 60s
- [ ] Green CI badge on main
- [ ] `pip install openfreqbench` installs cleanly

---

### v0.3.0 — Full Runner Hierarchy

**Target:** P1-01 through P1-05 closed.

**Definition of done:**
- [ ] Cache/resume works (`ofb status`, `ofb resume`)
- [ ] Worker parallelism works (`workers: 4`)
- [ ] Bonferroni and effect sizes in reports
- [ ] CVaR-95 in all summaries
- [ ] Transfer benchmark mode working

---

### v0.4.0 — Documentation and Developer Experience

**Target:** P1-06 through P1-10 closed.

**Definition of done:**
- [ ] `ofb new estimator` scaffolding works
- [ ] `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, `RESULT_SCHEMA.md` written and reviewed
- [ ] Full `CONTRIBUTING.md`
- [ ] All 18 scenarios ported and tested

---

### v1.0.0 — Public Launch

**Target:** All P0 and P1 closed; selected P2 tickets closed.

**Definition of done:**
- [ ] All v0.x milestones complete
- [ ] PyPI release: `pip install openfreqbench` installs and runs correctly
- [ ] Documentation published (MkDocs or similar)
- [ ] `CITATION.cff` written
- [ ] At least P2-01 (paired tests), P2-02 (ranking stability), P2-04 (Pareto), P2-06 (compliance) closed
- [ ] Release notes in `CHANGELOG.md`

---

## 6. Migration Phases Quick Reference

| Phase | Duration | Goal | Key Actions |
|---|---|---|---|
| **0 — Hygiene** | 1–2 days | Clean repo; CI restored | Gitignore artifacts; restore CI; create `src/openfreqbench/` |
| **1 — First Slice** | 3–5 days | One estimator + one scenario end-to-end | Port ZC + G1_E1 + matching tests; verify parity |
| **2 — Wire Runner** | 3–5 days | `ofb run` produces real RMSE | Extract validate_estimator into modules; replace stub |
| **3 — Port All** | 5–7 days | All 10+ estimators, 18 scenarios, all tests | Port one at a time; verify parity each time |
| **4 — Cache & Parallel** | 3–5 days | Resume works; multi-worker works | ArtifactIdentity; ProcessPoolExecutor |
| **5 — SRP Cleanup** | 3–5 days | Timing and tuning external to estimators | TimingHarness; TuningRunner |
| **6 — Stats Upgrades** | 3–5 days | Rigorous reports | Bonferroni; Cohen's d; CVaR-95 |
| **7 — Launch Prep** | 5–7 days | Public v1.0.0 | Docs; CI; PyPI; CITATION.cff |

**Total estimated time to v1.0.0:** 4–6 weeks of focused work.

---

## 7. Dependency Graph

```
P0-05 (rename package)
  └─► P0-02 (port estimators)
  └─► P0-03 (port scenarios)

P0-01 (fix runner stub)
  └─► P0-09 (YAML config)
      └─► P1-07 (transfer mode)
  └─► P0-02 (port estimators)
  └─► P0-03 (port scenarios)
  └─► P1-01 (cache key)
      └─► P1-04 (worker pool)
      └─► P1-08 (status/resume)

P0-06 (restore CI)
  └─► P0-10 (smoke tests)

P1-02 (extract TuningRunner)
  └─► P2-08 (Bayesian tuning)

P1-03 (extract TimingHarness)
  └─► P2-04 (Pareto figure)  [needs latency data]

P0-04 (port tests)
  [provides regression anchors for all other tickets]

Full suite results (G1–G4)
  └─► P2-03 (N-way ANOVA)
  └─► P2-02 (ranking stability)
```

---

## 8. Per-Module Work Summary

| Module | P0 Work | P1 Work | P2 Work |
|---|---|---|---|
| `estimators/_base.py` | Remove self-timing (P0-02) | Remove `optimize()` (P1-02); remove `step()` timing (P1-03) | — |
| `estimators/*.py` | Port 10+ estimators (P0-02) | — | — |
| `scenarios/_base.py` | Port from openfreqbench (P0-03) | — | — |
| `scenarios/g1/`–`g4/` | Port 18 scenarios (P0-03) | — | — |
| `scenarios/g5/` | — | — | OpenDSS (P2-09) |
| `runners/trace_runner.py` | New — wire real logic (P0-01) | Wire TimingHarness (P1-03) | — |
| `runners/scenario_method_runner.py` | New — from validate_estimator (P0-01) | Skip-existing (P1-01) | — |
| `runners/scenario_runner.py` | New (P0-01) | Wire TuningRunner (P1-02); worker dispatch (P1-04) | Transfer mode (P1-07) |
| `runners/suite_runner.py` | New (P0-01) | ProcessPoolExecutor (P1-04) | N-way ANOVA trigger (P2-03) |
| `core/run_identity.py` | — | ArtifactIdentity (P1-01) | — |
| `core/config.py` | Extend for YAML (P0-09) | TransferConfig (P1-07) | BayesianTuning (P2-08) |
| `tuning/grid_search.py` | — | Extract TuningRunner (P1-02) | — |
| `tuning/bayesian.py` | — | — | Bayesian opt (P2-08) |
| `profiling/timing.py` | — | TimingHarness (P1-03) | — |
| `profiling/memory.py` | — | Wire MemoryMeter (P1-03) | — |
| `metrics/frequency.py` | — | CVaR-95 (P1-10) | — |
| `metrics/compliance.py` | — | — | IEEE C37.118 (P2-06) |
| `stats/hypothesis.py` | Enforce n_runs ≥ 30 (P0-09) | Bonferroni (P1-05) | Paired tests (P2-01); ANOVA (P2-03) |
| `stats/effects.py` | — | Cohen's d (P1-05) | — |
| `stats/ranking.py` | — | — | Ranking stability (P2-02) |
| `io/artifact_store.py` | New — from validate_estimator (P0-01) | Identity-based paths (P1-01) | — |
| `reports/comparison.py` | New — from validate_estimator (P0-01) | Bonferroni in output (P1-05) | Effect sizes (P2-01) |
| `plotting/pareto.py` | — | — | Pareto frontier (P2-04) |
| `plotting/violin.py` | — | — | Violin+KDE (P2-05) |
| `cli/scaffold.py` | — | `ofb new estimator/scenario` (P1-06) | — |
| `cli/commands/status.py` | — | `ofb status` (P1-08) | — |
| `cli/commands/run.py` | Wire to SuiteRunner (P0-01, P0-09) | `ofb resume` (P1-08) | — |
| `tests/smoke/` | Add smoke suite (P0-10) | — | — |
| `tests/unit/` | Port openfreqbench tests (P0-04) | — | — |
| `tests/integration/` | Add CLI run test (P0-01) | Resume test (P1-08) | — |
| `tests/regression/` | Add parity tests (P0-02, P0-03) | — | — |
| `CONTRIBUTING.md` | Write minimal version (P0-08) | Full version (P1-09) | — |
| `ARCHITECTURE.md` | — | Write (P1-09) | — |
| `BENCHMARK_SPEC.md` | — | Write (P1-09) | Update for compliance (P2-06) |
| `RESULT_SCHEMA.md` | — | Write (P1-09) | — |

---

*End of ticket backlog and roadmap. File version 1.0 — 2026-03-18.*
