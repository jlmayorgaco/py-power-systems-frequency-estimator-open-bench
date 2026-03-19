# Contributing to OpenFreqBench

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/smoke/
```

## Adding an Estimator

1. Create `src/openfreqbench/estimators/<name>.py`
2. Subclass `BaseEstimator`, implement `_step`, `reset`, `latency_samples`
3. Register in `core/registry.py`
4. Add tests in `tests/unit/estimators/test_<name>.py`

## Adding a Scenario

1. Create `src/openfreqbench/scenarios/<group>/<name>.py`
2. Subclass `ScenarioBase` as `@dataclass`, implement `build()`
3. Register in `core/registry.py`
4. Add tests in `tests/unit/scenarios/test_<name>.py`

## Code Style

- `ruff check` for linting
- `mypy` for type checks
- `pytest` for all tests

## Design Rules

- `estimators/`, `scenarios/`, `metrics/` → numpy only
- No timing inside `_step()`
- No fake/hardcoded metric values
