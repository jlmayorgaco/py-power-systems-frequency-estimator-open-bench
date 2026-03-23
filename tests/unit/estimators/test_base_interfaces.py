"""
Unit tests for the estimator base interfaces.

Covers:
  - EstimatorOutput dataclass
  - EstimatorSpec dataclass
  - TuningSpec.candidate_grid() correctness
  - BaseEstimator.update() on a trivial estimator
  - EstimatorRegistry has all expected built-ins
  - EstimatorRegistry.list_by_family() grouping
"""

from __future__ import annotations

import math

import numpy as np
from openfreqbench.estimators._base import TuningParam, TuningSpec
from openfreqbench.estimators._outputs import EstimatorOutput, EstimatorSpec
from openfreqbench.estimators.registry import EstimatorRegistry
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# EstimatorOutput
# ─────────────────────────────────────────────────────────────────────────────


def test_estimator_output_valid_finite():
    o = EstimatorOutput(frequency_hz=60.0, valid=True)
    assert o.is_finite
    assert o.is_valid


def test_estimator_output_invalid_nan():
    o = EstimatorOutput(frequency_hz=float("nan"), valid=True)
    assert not o.is_finite
    assert not o.is_valid


def test_estimator_output_valid_false():
    o = EstimatorOutput(frequency_hz=60.0, valid=False)
    assert o.is_finite
    assert not o.is_valid


def test_estimator_output_defaults():
    o = EstimatorOutput(frequency_hz=60.0)
    assert math.isnan(o.rocof_hz_s)
    assert math.isnan(o.phase_rad)
    assert math.isnan(o.amplitude_pu)
    assert o.valid is True


def test_estimator_output_to_dict():
    o = EstimatorOutput(frequency_hz=60.0, valid=True, rocof_hz_s=0.5)
    d = o.to_dict()
    assert d["frequency_hz"] == 60.0
    assert d["rocof_hz_s"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# EstimatorSpec
# ─────────────────────────────────────────────────────────────────────────────


def test_estimator_spec_frozen():
    s = EstimatorSpec(
        name="Test",
        family="TestFam",
        family_path="test/path",
        complexity="O(1)",
        latency_type="causal",
    )
    with pytest.raises((AttributeError, TypeError)):
        s.name = "Other"  # type: ignore[misc]


def test_estimator_spec_to_dict():
    s = EstimatorSpec(
        name="X",
        family="Y",
        family_path="z/path",
        complexity="O(N)",
        latency_type="semi-causal",
    )
    d = s.to_dict()
    assert d["name"] == "X"
    assert d["family_path"] == "z/path"
    assert d["nominal_freq_hz"] == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# TuningSpec.candidate_grid()
# ─────────────────────────────────────────────────────────────────────────────


def test_tuning_spec_empty_returns_single_empty_dict():
    ts = TuningSpec(params=[], objective="RMSE_HZ")
    grid = ts.candidate_grid()
    assert grid == [{}]


def test_tuning_spec_single_param_grid():
    ts = TuningSpec(
        params=[
            TuningParam(name="k", default=1.0, type="float", values=[0.5, 1.0, 2.0]),
        ],
    )
    grid = ts.candidate_grid()
    assert len(grid) == 3
    assert {"k": 0.5} in grid
    assert {"k": 2.0} in grid


def test_tuning_spec_cartesian_product():
    ts = TuningSpec(
        params=[
            TuningParam("a", 1, "int", values=[1, 2]),
            TuningParam("b", 10, "int", values=[10, 20, 30]),
        ],
    )
    grid = ts.candidate_grid()
    assert len(grid) == 6
    assert {"a": 1, "b": 10} in grid
    assert {"a": 2, "b": 30} in grid


def test_tuning_spec_n_candidates():
    ts = TuningSpec(
        params=[
            TuningParam("a", 1, "int", values=[1, 2]),
            TuningParam("b", 10, "int", values=[10, 20, 30]),
        ],
    )
    assert ts.n_candidates() == 6


# ─────────────────────────────────────────────────────────────────────────────
# BaseEstimator.update() on all built-ins
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", EstimatorRegistry.list_names())
def test_all_registered_update_returns_output(name):
    est = EstimatorRegistry.build(name)
    est._fs_hint = 10_000.0
    est.reset()
    result = est.update(0.5, timestamp=0.0)
    assert isinstance(result, EstimatorOutput)
    assert math.isfinite(result.frequency_hz) or not result.valid


@pytest.mark.parametrize("name", EstimatorRegistry.list_names())
def test_all_registered_run_no_crash(name):
    est = EstimatorRegistry.build(name)
    est._fs_hint = 10_000.0
    t = np.arange(500) / 10_000.0
    v = np.sin(2 * np.pi * 60.0 * t)
    out = est.run(v)
    assert len(out) == len(v)
    assert all(np.isfinite(out))


# ─────────────────────────────────────────────────────────────────────────────
# EstimatorRegistry
# ─────────────────────────────────────────────────────────────────────────────


def test_registry_contains_all_expected_names():
    names = set(EstimatorRegistry.list_names())
    expected = {
        "Baseline_Passthrough",
        "SOGI_FLL",
        "EKF_Freq",
        "FFTPeak",
        "IpDFT",
        "ZeroCrossing",
        "RDFT",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


def test_registry_list_by_family():
    groups = EstimatorRegistry.list_by_family()
    # All built-in families should be present
    assert any("f0_pll" in fp for fp in groups)
    assert any("f1_kalman" in fp for fp in groups)
    assert any("f2_window" in fp for fp in groups)
    assert any("f3_recursive" in fp for fp in groups)


def test_registry_descriptors_all_complete():
    descs = EstimatorRegistry.list_descriptors()
    required = {
        "name",
        "family",
        "family_path",
        "complexity",
        "latency_type",
        "suggested_objective",
        "tuning_params",
    }
    for d in descs:
        assert required.issubset(set(d.keys())), (
            f"Descriptor for {d.get('name')} missing keys: {required - set(d.keys())}"
        )
