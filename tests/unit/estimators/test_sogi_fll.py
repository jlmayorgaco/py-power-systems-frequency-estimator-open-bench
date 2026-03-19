"""
Unit tests for SOGIFLLEstimator (monophasic/f0_pll/sogi_fll.py)

Covers:
  - instantiation and reset
  - latency_samples is positive
  - update() returns EstimatorOutput
  - run() returns same-length output
  - no NaN on clean 60 Hz signal after settling
  - frequency settles near 60 Hz on steady-state input
  - tuning_spec() returns TuningSpec with params
  - descriptor() returns required keys
  - spec() returns EstimatorSpec
  - update() valid flag True after warm-up
  - update() valid flag False before warm-up
  - set_params triggers reset
  - amplitude_pu in output
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openfreqbench.estimators.monophasic.f0_pll.sogi_fll import SOGIFLLEstimator
from openfreqbench.estimators._outputs import EstimatorOutput, EstimatorSpec
from openfreqbench.estimators._base import TuningSpec


FS  = 10_000.0
F0  = 60.0
T_S = 0.5
N   = int(FS * T_S)


def _sine(fs=FS, f0=F0, n=N, A=1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return A * np.sin(2 * np.pi * f0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# Instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiate_default():
    est = SOGIFLLEstimator()
    assert est is not None


def test_name():
    assert SOGIFLLEstimator.NAME == "SOGI_FLL"


def test_family_path():
    assert "f0_pll" in SOGIFLLEstimator.FAMILY_PATH


def test_latency_positive():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    assert est.latency_samples > 0


# ─────────────────────────────────────────────────────────────────────────────
# update() interface
# ─────────────────────────────────────────────────────────────────────────────

def test_update_returns_estimator_output():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    result = est.update(0.5, timestamp=0.0)
    assert isinstance(result, EstimatorOutput)


def test_update_valid_false_before_warmup():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    result = est.update(0.0, timestamp=0.0)
    assert result.valid is False


def test_update_valid_true_after_warmup():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    v   = _sine()
    Ts  = 1.0 / FS
    lat = est.latency_samples
    for i, x in enumerate(v):
        result = est.update(float(x), timestamp=i * Ts)
    assert result.valid is True


def test_update_amplitude_pu_set():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    v = _sine(A=1.0)
    result = None
    for x in v:
        result = est.update(float(x))
    assert math.isfinite(result.amplitude_pu)
    assert result.amplitude_pu > 0.0


def test_update_phase_rad_set():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    v = _sine()
    result = None
    for x in v:
        result = est.update(float(x))
    assert math.isfinite(result.phase_rad)


# ─────────────────────────────────────────────────────────────────────────────
# run() batch interface
# ─────────────────────────────────────────────────────────────────────────────

def test_run_output_length():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    v   = _sine()
    out = est.run(v)
    assert len(out) == len(v)


def test_run_no_nan_after_settling():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    v   = _sine()
    out = est.run(v)
    L   = est.latency_samples
    assert np.all(np.isfinite(out[L:])), "NaN/inf after latency window"


def test_run_frequency_settles_near_60hz():
    """After warm-up, mean frequency estimate < 1.0 Hz from 60 Hz."""
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    v   = _sine()
    out = est.run(v)
    L   = est.latency_samples + int(0.1 * N)
    valid = out[L:]
    valid = valid[np.isfinite(valid)]
    assert len(valid) > 10
    err = abs(float(np.mean(valid)) - F0)
    assert err < 1.0, f"Mean freq error {err:.4f} Hz after settling"


# ─────────────────────────────────────────────────────────────────────────────
# Self-description API
# ─────────────────────────────────────────────────────────────────────────────

def test_spec_returns_estimator_spec():
    s = SOGIFLLEstimator.spec()
    assert isinstance(s, EstimatorSpec)


def test_spec_name():
    s = SOGIFLLEstimator.spec()
    assert s.name == "SOGI_FLL"


def test_tuning_spec_returns_tuning_spec():
    ts = SOGIFLLEstimator.tuning_spec()
    assert isinstance(ts, TuningSpec)


def test_tuning_spec_has_params():
    ts = SOGIFLLEstimator.tuning_spec()
    assert len(ts.params) >= 1


def test_tuning_spec_candidate_grid_nonempty():
    ts = SOGIFLLEstimator.tuning_spec()
    grid = ts.candidate_grid()
    assert len(grid) >= 1
    assert isinstance(grid[0], dict)


def test_descriptor_required_keys():
    desc = SOGIFLLEstimator.descriptor()
    for key in ("name", "family", "family_path", "complexity",
                "latency_type", "suggested_objective", "tuning_params"):
        assert key in desc, f"Missing descriptor key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# set_params / reset
# ─────────────────────────────────────────────────────────────────────────────

def test_set_params_updates_gamma():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    est.set_params(gamma=200.0)
    assert est._gamma == pytest.approx(200.0)


def test_set_params_triggers_reset():
    est = SOGIFLLEstimator()
    est._fs_hint = FS
    est.reset()
    # Run a few samples to change internal state
    v = _sine(n=100)
    for x in v:
        est.step(x)
    alpha_before = est._alpha
    est.set_params(k=2.0)
    assert est._alpha == 0.0   # reset to initial state
