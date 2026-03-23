"""
Unit tests for openfreqbench.stats.aggregate

Covers:
  - aggregate_monte_carlo returns correct keys
  - ordering: p5 <= p25 <= p50 <= p75 <= p95
  - mean/median coherent with known data
  - CoV correct for deterministic sequence
  - bootstrap CI: lo <= mean <= hi
  - n, n_valid, n_nan counts
  - empty input returns {}
  - all-NaN values produce n_valid=0, no crash
  - compare_estimators returns expected structure
"""

from __future__ import annotations

import math

import numpy as np
from openfreqbench.stats.aggregate import aggregate_monte_carlo, compare_estimators
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_run(rmse: float, tps: float = 1.0, seed: int = 0) -> dict:
    """Minimal metric dict matching compute_metrics() output structure."""
    return {
        "RMSE_HZ": {"name": "RMSE_HZ", "value": rmse, "units": "Hz"},
        "TIME_PER_SAMPLE_US": {"name": "TIME_PER_SAMPLE_US", "value": tps, "units": "us"},
    }


def _make_runs(rmse_values, tps_values=None):
    tps = tps_values or [1.0] * len(rmse_values)
    return [_make_run(r, t) for r, t in zip(rmse_values, tps)]


# ─────────────────────────────────────────────────────────────────────────────
# Basic structure tests
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_returns_empty():
    assert aggregate_monte_carlo([]) == {}


def test_expected_keys_present():
    runs = _make_runs([0.01, 0.02, 0.03, 0.04, 0.05])
    agg = aggregate_monte_carlo(runs)
    block = agg["RMSE_HZ"]
    required = {
        "n",
        "n_valid",
        "n_nan",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "cov",
        "ci95_lo",
        "ci95_hi",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    }
    assert required.issubset(set(block.keys())), f"Missing keys: {required - set(block.keys())}"


# ─────────────────────────────────────────────────────────────────────────────
# Count tests
# ─────────────────────────────────────────────────────────────────────────────


def test_n_counts():
    runs = _make_runs([0.01, 0.02, 0.03])
    agg = aggregate_monte_carlo(runs)
    b = agg["RMSE_HZ"]
    assert b["n"] == 3
    assert b["n_valid"] == 3
    assert b["n_nan"] == 0


def test_n_nan_counted():
    # Inject NaN via non-finite value (non-finite values are excluded)
    runs = [
        {"RMSE_HZ": {"value": 0.01, "name": "RMSE_HZ", "units": "Hz"}},
        {"RMSE_HZ": {"value": float("nan"), "name": "RMSE_HZ", "units": "Hz"}},
        {"RMSE_HZ": {"value": 0.03, "name": "RMSE_HZ", "units": "Hz"}},
    ]
    agg = aggregate_monte_carlo(runs)
    b = agg["RMSE_HZ"]
    assert b["n"] == 3
    assert b["n_valid"] == 2
    assert b["n_nan"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Percentile ordering
# ─────────────────────────────────────────────────────────────────────────────


def test_percentile_ordering():
    rng = np.random.default_rng(42)
    vals = rng.uniform(0.001, 0.1, 50).tolist()
    runs = _make_runs(vals)
    agg = aggregate_monte_carlo(runs)
    b = agg["RMSE_HZ"]
    assert b["p1"] <= b["p5"] <= b["p25"] <= b["p50"] <= b["p75"] <= b["p95"] <= b["p99"], (
        f"Percentile ordering violated: {b}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Known-value tests
# ─────────────────────────────────────────────────────────────────────────────


def test_mean_exact():
    vals = [0.010, 0.020, 0.030, 0.040, 0.050]
    agg = aggregate_monte_carlo(_make_runs(vals))
    assert abs(agg["RMSE_HZ"]["mean"] - np.mean(vals)) < 1e-12


def test_median_exact():
    vals = [0.010, 0.020, 0.030, 0.040, 0.050]
    agg = aggregate_monte_carlo(_make_runs(vals))
    assert abs(agg["RMSE_HZ"]["median"] - np.median(vals)) < 1e-12


def test_min_max_exact():
    vals = [0.010, 0.020, 0.030, 0.040, 0.050]
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    assert abs(b["min"] - 0.010) < 1e-12
    assert abs(b["max"] - 0.050) < 1e-12


def test_cov_correct():
    # Constant sequence: std=0 → CoV=0
    vals = [0.050] * 20
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    assert abs(b["std"]) < 1e-12
    assert abs(b["cov"]) < 1e-10


def test_cov_nonzero():
    vals = [0.010 * i for i in range(1, 11)]  # linear spread
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    assert b["cov"] > 0.0
    expected_cov = float(np.std(vals, ddof=1) / abs(np.mean(vals)))
    assert abs(b["cov"] - expected_cov) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────


def test_bootstrap_ci_bounds_contain_mean():
    rng = np.random.default_rng(7)
    vals = rng.normal(0.05, 0.01, 30).tolist()
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    mean = b["mean"]
    # CI should contain the empirical mean with high probability
    assert b["ci95_lo"] <= mean + 1e-9, f"CI lower bound {b['ci95_lo']} > mean {mean}"
    assert b["ci95_hi"] >= mean - 1e-9, f"CI upper bound {b['ci95_hi']} < mean {mean}"


def test_bootstrap_ci_width_positive():
    vals = np.random.default_rng(99).normal(0.05, 0.01, 30).tolist()
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    assert b["ci95_hi"] > b["ci95_lo"]


def test_bootstrap_ci_collapses_for_constant():
    # For a constant distribution, CI should be very narrow
    vals = [0.05] * 30
    agg = aggregate_monte_carlo(_make_runs(vals))
    b = agg["RMSE_HZ"]
    width = b["ci95_hi"] - b["ci95_lo"]
    assert width < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# All-NaN input
# ─────────────────────────────────────────────────────────────────────────────


def test_all_nan_no_crash():
    runs = [
        {"RMSE_HZ": {"value": float("nan"), "name": "RMSE_HZ", "units": "Hz"}} for _ in range(5)
    ]
    agg = aggregate_monte_carlo(runs)
    b = agg["RMSE_HZ"]
    assert b["n_valid"] == 0
    assert b["n"] == 5
    assert math.isnan(b["mean"])


# ─────────────────────────────────────────────────────────────────────────────
# Multiple metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_metrics_independent():
    rmse_vals = [0.01, 0.02, 0.03]
    tps_vals = [1.0, 2.0, 3.0]
    runs = _make_runs(rmse_vals, tps_vals)
    agg = aggregate_monte_carlo(runs)
    assert "RMSE_HZ" in agg
    assert "TIME_PER_SAMPLE_US" in agg
    assert abs(agg["TIME_PER_SAMPLE_US"]["mean"] - 2.0) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# compare_estimators
# ─────────────────────────────────────────────────────────────────────────────


def test_compare_estimators_structure():
    runs_a = _make_runs([0.010, 0.011, 0.012, 0.013, 0.014])
    runs_b = _make_runs([0.050, 0.051, 0.052, 0.053, 0.054])
    cmp = compare_estimators(runs_a, runs_b, metric="RMSE_HZ")
    if "error" in cmp:
        pytest.skip(f"scipy not available: {cmp['error']}")
    required = {
        "metric",
        "mean_a",
        "mean_b",
        "delta_mean",
        "t_stat",
        "p_value",
        "significant",
        "n_a",
        "n_b",
    }
    assert required.issubset(set(cmp.keys()))


def test_compare_estimators_direction():
    runs_a = _make_runs([0.010] * 20)
    runs_b = _make_runs([0.050] * 20)
    cmp = compare_estimators(runs_a, runs_b, metric="RMSE_HZ")
    if "error" in cmp:
        pytest.skip("scipy not available")
    assert cmp["delta_mean"] < 0  # a is better (lower RMSE)
    assert cmp["significant"] is True
