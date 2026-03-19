# OpenFreqBench — Testing Guide
> Branch: `q1-arch-final` | Tests: 232 passing | Updated: 2026-03-19

---

## Test Levels

| Level | Command | Duration | Gate |
|-------|---------|----------|------|
| Smoke | `pytest tests/smoke/` | < 30s | Every commit |
| Unit | `pytest tests/unit/` | < 2min | Every PR |
| Integration | `pytest tests/integration/` | < 5min | Every PR |
| Regression | `pytest tests/regression/` | ~10min | Before merge to main |
| Full | `pytest tests/` | ~15min | Before release |

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Quick sanity check (< 30s)
pytest tests/smoke/ -v

# Per-class isolation
pytest tests/unit/ -v

# Cross-layer pipelines
pytest tests/integration/ -v

# Full suite
pytest tests/ -v

# Single test file
pytest tests/unit/estimators/test_ekf_freq.py -v

# By marker
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=openfreqbench --cov-report=term-missing
```

---

## Test Philosophy

1. **No mocking of numpy or estimator internals.**
   Every test runs real signal processing on real numpy arrays.
   Mock only I/O boundaries (file system, network) when unavoidable.

2. **Smoke tests use the real pipeline, just short.**
   `T_s = 0.1s`, `n_runs = 2` — same code path as a real benchmark run.

3. **Unit tests are deterministic.**
   Every test that uses a random signal passes `seed=42` explicitly.
   `np.random.default_rng(42)` — never `np.random.seed()`.

4. **Regression tests compare against `v1/PMU/pfebench/` reference values.**
   Reference JSON files are stored in `tests/fixtures/reference/`.
   Tolerance: RMSE difference < 0.1 Hz (1% of nominal 60 Hz).

5. **Known-answer tests for all metrics.**
   Every metric function has at least one test with an analytically computable answer.
   E.g.: constant offset signal → RMSE = |offset| exactly.

6. **Each real estimator requires these tests (minimum):**
   - `test_instantiate_with_default_config`
   - `test_reset_restores_initial_state`
   - `test_step_returns_estimator_output`
   - `test_step_output_no_nan`
   - `test_step_output_in_valid_range` (40 ≤ f ≤ 80 Hz)
   - `test_pure_60hz_converges` (output ≈ 60 Hz ± tolerance after warm-up)
   - `test_mc_reproducible_with_seed` (same seed → same output array)

7. **Each real scenario requires these tests (minimum):**
   - `test_build_returns_scenario_output`
   - `test_output_shapes_consistent` (all arrays same length)
   - `test_f_true_in_valid_range`
   - `test_v_amplitude_reasonable` (0.3 ≤ A_rms ≤ 1.5 pu)
   - `test_deterministic_with_seed`
   - `test_different_seeds_produce_different_noise`

---

## Test File Map

```
tests/
├── conftest.py                    # Shared fixtures: tmp_path, waveforms, configs
├── fixtures/
│   ├── waveforms.py               # Pure 60 Hz, step, ramp, noisy sine generators
│   ├── configs.py                 # Minimal BenchmarkConfig for tests
│   ├── tempdirs.py                # Temporary artifact directories
│   └── reference/                 # Reference JSON from v1/PMU/pfebench/ runs
│
├── smoke/                         # < 30s, run on every commit
│   ├── test_imports.py            # All public modules import without error
│   ├── test_quick_run.py          # One-shot ZeroCrossing × G1_E1, n_runs=2
│   └── test_cli_doctor.py         # `ofb doctor` exits 0
│
├── unit/
│   ├── estimators/
│   │   ├── test_base_interfaces.py       # All registered estimators satisfy contract
│   │   ├── test_baseline_passthrough.py  # BaselinePassthrough unit tests
│   │   ├── test_zero_crossing.py         # ZeroCrossingEstimator unit tests
│   │   ├── test_ekf_freq.py              # EKFFreqEstimator unit tests (+ stability)
│   │   ├── test_fft_peak.py              # FFTPeakEstimator unit tests
│   │   ├── test_ipdft.py                 # IpDFTEstimator unit tests
│   │   ├── test_rdft.py                  # RDFTEstimator unit tests
│   │   └── test_sogi_fll.py             # SOGIFLLEstimator unit tests
│   │
│   ├── scenarios/
│   │   ├── test_g1_e1_pure_60hz.py       # Build, shapes, physics
│   │   └── test_g2_e2_freq_ramp.py       # Build, f_true profile correct
│   │
│   ├── metrics/
│   │   └── test_frequency.py             # compute_metrics known-answer tests
│   │
│   ├── stats/
│   │   └── test_aggregate.py             # aggregate_monte_carlo statistics
│   │
│   ├── profiling/
│   │   └── test_timing.py                # TimingHarness measures real time
│   │
│   ├── core/
│   │   └── test_run_identity.py          # SHA-256 cache key determinism
│   │
│   └── test_checkpoint.py               # 26 tests covering full lifecycle
│
├── integration/
│   ├── test_trace_runner.py              # TraceRunner: scenario + estimator → metrics
│   ├── test_scenario_method_runner.py    # ScenarioMethodRunner: MC loop → aggregates
│   ├── test_cli_run.py                   # `ofb run` e2e with CliRunner
│   └── test_artifact_store.py            # JSON save/load, cache key collision test
│
└── regression/
    ├── test_legacy_parity_zero_crossing.py  # ZC RMSE matches pfebench reference ±0.1 Hz
    └── test_reference_metrics.py            # compute_metrics output matches reference JSON
```

---

## Writing a New Estimator Test

```python
# tests/unit/estimators/test_my_estimator.py
import numpy as np
import pytest
from openfreqbench.estimators.monophasic.f1_kalman.my_estimator import MyEstimator

FS = 10_000.0
F_NOM = 60.0


@pytest.fixture
def est():
    return MyEstimator(config={"fs": FS})


def test_instantiate_with_default_config():
    e = MyEstimator()
    assert e._config["fs"] == FS


def test_reset_restores_initial_state(est):
    # Run some samples to change state
    for _ in range(100):
        est.update(np.sin(2 * np.pi * F_NOM * _ / FS))
    est.reset()
    # State is equivalent to fresh construction
    est2 = MyEstimator(config=est._config.copy())
    assert est._f_est == est2._f_est


def test_step_output_no_nan(est):
    t = np.linspace(0, 0.5, int(FS * 0.5))
    v = np.sin(2 * np.pi * F_NOM * t)
    for sample in v:
        out = est.update(float(sample))
        assert np.isfinite(out.frequency_hz), f"NaN output after {_} samples"


def test_step_output_in_valid_range(est):
    t = np.linspace(0, 0.5, int(FS * 0.5))
    v = np.sin(2 * np.pi * F_NOM * t)
    for sample in v:
        out = est.update(float(sample))
        assert 40.0 <= out.frequency_hz <= 80.0


def test_pure_60hz_converges(est):
    """After warm-up, output must track 60 Hz within tolerance."""
    t = np.linspace(0, 1.0, int(FS * 1.0))
    v = np.sin(2 * np.pi * F_NOM * t)
    out = est.run(v)
    warm_up = est.structural_latency_samples()
    rmse = float(np.sqrt(np.mean((out[warm_up:] - F_NOM) ** 2)))
    assert rmse < 0.5, f"RMSE {rmse:.4f} Hz exceeds 0.5 Hz on pure 60 Hz"


def test_mc_reproducible_with_seed():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.5, int(FS * 0.5))
    v = np.sin(2 * np.pi * F_NOM * t) + rng.normal(0, 0.01, len(t))

    e1 = MyEstimator(config={"fs": FS})
    e2 = MyEstimator(config={"fs": FS})
    out1 = e1.run(v.copy())
    out2 = e2.run(v.copy())
    np.testing.assert_array_equal(out1, out2)


def test_spec_name():
    assert MyEstimator.SPEC.name == "MyEst"


def test_tuning_spec_bounds_valid():
    for param in MyEstimator.tuning_spec().params:
        if param.range:
            lo, hi, n = param.range
            assert lo < hi
            assert n >= 2
```

---

## Writing a New Scenario Test

```python
# tests/unit/scenarios/test_my_scenario.py
import numpy as np
import pytest
from openfreqbench.scenarios.g2.my_scenario import MyScenario


@pytest.fixture
def scen():
    return MyScenario(fs_hz=10_000.0, T_s=2.0)


def test_build_returns_output(scen):
    out = scen.build()
    assert out.v is not None
    assert out.f_true is not None
    assert out.t is not None


def test_shapes_consistent(scen):
    out = scen.build()
    assert len(out.v) == len(out.f_true) == len(out.t)
    expected_len = int(scen.fs_hz * scen.T_s)
    assert len(out.v) == expected_len


def test_f_true_in_valid_range(scen):
    out = scen.build()
    assert np.all(out.f_true >= 40.0)
    assert np.all(out.f_true <= 80.0)


def test_deterministic_with_seed(scen):
    out1 = scen.build()
    out2 = scen.build()
    np.testing.assert_array_equal(out1.v, out2.v)


def test_different_seeds_differ():
    s1 = MyScenario(seed=0)
    s2 = MyScenario(seed=1)
    assert not np.allclose(s1.build().v, s2.build().v)
```

---

## Known-Answer Tests (Metrics)

```python
# tests/unit/metrics/test_frequency.py

def test_rmse_constant_offset():
    f_hat  = np.full(10000, 60.1)
    f_true = np.full(10000, 60.0)
    metrics = compute_metrics(f_hat, f_true, exec_time_s=0.001,
                               latency_samples=0, cfg=cfg, scenario_id="test")
    assert abs(metrics["RMSE_HZ"]["value"] - 0.1) < 1e-10


def test_rmse_zero_error():
    f = np.full(10000, 60.0)
    metrics = compute_metrics(f, f.copy(), 0.001, 0, cfg, "test")
    assert metrics["RMSE_HZ"]["value"] == 0.0


def test_trip_time_zero_when_always_below_threshold():
    f_hat  = np.full(10000, 60.3)   # error = 0.3 < 0.5 threshold
    f_true = np.full(10000, 60.0)
    metrics = compute_metrics(f_hat, f_true, 0.001, 0, cfg, "test")
    assert metrics["TRIP_TIME_0p5_S"]["value"] == 0.0


def test_trip_time_full_when_always_above():
    f_hat  = np.full(10000, 61.0)   # error = 1.0 > 0.5 threshold
    f_true = np.full(10000, 60.0)
    metrics = compute_metrics(f_hat, f_true, 0.001, 0, cfg, "test")
    # 10000 samples at 10 kHz = 1.0s total
    assert abs(metrics["TRIP_TIME_0p5_S"]["value"] - 1.0) < 1e-6
```

---

## CI Integration (Phase 2 — restore GitHub Actions)

```yaml
# .github/workflows/ci.yml (to be restored in Phase 2)
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python }}"}
      - run: pip install -e ".[dev]"
      - run: pytest tests/smoke/ tests/unit/ tests/integration/ -v
      - run: ruff check src/
      - run: mypy src/openfreqbench/ --ignore-missing-imports
```

---

## Coverage Goals

| Module | Current | Target (M1) | Target (publication) |
|--------|---------|-------------|---------------------|
| `estimators/common/` | ~90% | 95% | 95% |
| `estimators/monophasic/` | ~85% | 90% | 90% |
| `scenarios/` | ~80% | 85% | 85% |
| `metrics/` | ~75% | 85% | 90% |
| `stats/` | ~40% | 70% | 90% |
| `runners/` | ~70% | 80% | 85% |
| `io/` | ~65% | 80% | 85% |
| `cli/` | ~60% | 70% | 75% |

Run coverage:
```bash
pytest tests/ --cov=openfreqbench --cov-report=html
open htmlcov/index.html
```

---

## Regression Test Protocol

When porting an estimator from `v1/PMU/pfebench/`:

1. Run the estimator in `v1/PMU/pfebench/` and save its output:
   ```python
   # In v1/PMU/pfebench/
   result = estimator.run_mc(scenario, n_seeds=30)
   json.dump(result, open("tests/fixtures/reference/ZeroCrossing_G1_E1.json", "w"))
   ```

2. Write the regression test:
   ```python
   def test_parity_with_pfebench_reference():
       ref = json.load(open("tests/fixtures/reference/ZeroCrossing_G1_E1.json"))
       est = ZeroCrossingEstimator(config={"fs": 10000.0})
       ...
       assert abs(rmse - ref["RMSE_HZ_mean"]) < 0.1  # ±0.1 Hz tolerance
   ```

3. Tolerance: `|new_rmse - ref_rmse| < 0.1 Hz` for steady-state scenarios,
   `< 0.5 Hz` for dynamic scenarios (step/ramp/phase jump).
