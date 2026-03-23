"""
Unit tests for EKFFreqEstimator (monophasic/f1_kalman/ekf_freq.py)

Covers:
  - instantiation and reset
  - latency_samples positive
  - update() returns EstimatorOutput
  - run() returns correct length
  - no NaN after settling
  - frequency tracks 60 Hz within 1 Hz
  - phase_rad is finite after update
  - spec() / tuning_spec() / descriptor() API
  - set_params triggers reset
"""

from __future__ import annotations

import math

import numpy as np
from openfreqbench.estimators._base import TuningSpec
from openfreqbench.estimators._outputs import EstimatorOutput, EstimatorSpec
from openfreqbench.estimators.monophasic.f1_kalman.ekf_freq import EKFFreqEstimator
import pytest

FS = 10_000.0
F0 = 60.0
T_S = 0.5
N = int(FS * T_S)


def _sine(fs=FS, f0=F0, n=N, A=1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return A * np.sin(2 * np.pi * f0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# Instantiation
# ─────────────────────────────────────────────────────────────────────────────


def test_instantiate_default():
    est = EKFFreqEstimator()
    assert est is not None


def test_name():
    assert EKFFreqEstimator.NAME == "EKF_Freq"


def test_family_path():
    assert "f1_kalman" in EKFFreqEstimator.FAMILY_PATH


def test_latency_positive():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    assert est.latency_samples > 0


# ─────────────────────────────────────────────────────────────────────────────
# update() interface
# ─────────────────────────────────────────────────────────────────────────────


def test_update_returns_estimator_output():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    result = est.update(0.5)
    assert isinstance(result, EstimatorOutput)


def test_update_phase_rad_finite():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    v = _sine()
    result = None
    for x in v:
        result = est.update(float(x))
    assert math.isfinite(result.phase_rad)


def test_update_amplitude_pu_positive():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    v = _sine(A=1.0)
    result = None
    for x in v:
        result = est.update(float(x))
    assert result.amplitude_pu > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# run() batch interface
# ─────────────────────────────────────────────────────────────────────────────


def test_run_output_length():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    v = _sine()
    out = est.run(v)
    assert len(out) == len(v)


def test_run_no_nan_after_latency():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    v = _sine()
    out = est.run(v)
    L = est.latency_samples
    assert np.all(np.isfinite(out[L:])), "NaN/inf after latency"


def test_steady_state_frequency_accuracy():
    """Mean frequency error < 1 Hz after convergence on 60 Hz signal."""
    est = EKFFreqEstimator()
    est._fs_hint = FS
    v = _sine()
    out = est.run(v)
    L = est.latency_samples + int(0.1 * N)
    valid = out[L:]
    valid = valid[np.isfinite(valid)]
    assert len(valid) > 10
    err = abs(float(np.mean(valid)) - F0)
    assert err < 1.0, f"Mean error {err:.4f} Hz"


def test_handles_zeros():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    v = np.zeros(N)
    out = est.run(v)
    assert all(np.isfinite(out))


# ─────────────────────────────────────────────────────────────────────────────
# Self-description API
# ─────────────────────────────────────────────────────────────────────────────


def test_spec_type():
    assert isinstance(EKFFreqEstimator.spec(), EstimatorSpec)


def test_tuning_spec_type():
    assert isinstance(EKFFreqEstimator.tuning_spec(), TuningSpec)


def test_tuning_spec_has_q_omega():
    names = {p.name for p in EKFFreqEstimator.tuning_ranges()}
    assert "q_omega" in names


def test_descriptor_keys():
    desc = EKFFreqEstimator.descriptor()
    for key in (
        "name",
        "family",
        "family_path",
        "complexity",
        "latency_type",
        "suggested_objective",
        "tuning_params",
    ):
        assert key in desc


# ─────────────────────────────────────────────────────────────────────────────
# set_params / reset
# ─────────────────────────────────────────────────────────────────────────────


def test_set_params_q_omega():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    est.set_params(q_omega=10.0)
    assert est._q_w == pytest.approx(10.0)


def test_set_params_r():
    est = EKFFreqEstimator()
    est._fs_hint = FS
    est.reset()
    est.set_params(r=1.0)
    assert est._r == pytest.approx(1.0)
