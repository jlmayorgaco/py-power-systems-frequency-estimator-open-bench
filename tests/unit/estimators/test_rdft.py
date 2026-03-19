"""
Unit tests for RDFTEstimator (monophasic/f3_recursive/rdft.py)

Covers:
  - instantiation and reset
  - latency = window_size
  - run() returns correct length
  - no NaN after settling on clean signal
  - frequency accuracy < 2 Hz on steady-state
  - spec() / tuning_spec() / descriptor() API
  - set_params triggers reset
"""
from __future__ import annotations

import numpy as np
import pytest

from openfreqbench.estimators.monophasic.f3_recursive.rdft import RDFTEstimator
from openfreqbench.estimators._outputs import EstimatorSpec
from openfreqbench.estimators._base import TuningSpec


FS  = 10_000.0
F0  = 60.0
N   = int(FS * 1.0)


def _sine(f0=F0, n=N, fs=FS) -> np.ndarray:
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# Instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiate_default():
    assert RDFTEstimator() is not None


def test_name():
    assert RDFTEstimator.NAME == "RDFT"


def test_family_path():
    assert "f3_recursive" in RDFTEstimator.FAMILY_PATH


@pytest.mark.parametrize("ws", [64, 128, 256, 512])
def test_latency_equals_window_size(ws):
    est = RDFTEstimator(params={"window_size": ws})
    est._fs_hint = FS
    est.reset()
    assert est.latency_samples == ws


# ─────────────────────────────────────────────────────────────────────────────
# run() shape
# ─────────────────────────────────────────────────────────────────────────────

def test_run_output_length():
    est = RDFTEstimator()
    est._fs_hint = FS
    out = est.run(_sine())
    assert len(out) == N


# ─────────────────────────────────────────────────────────────────────────────
# NaN / accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_no_nan_clean_signal():
    est = RDFTEstimator(params={"window_size": 256})
    est._fs_hint = FS
    out = est.run(_sine())
    assert np.all(np.isfinite(out))


def test_steady_state_accuracy():
    """Mean frequency error < 2 Hz after settling on 60 Hz signal."""
    est = RDFTEstimator(params={"window_size": 256, "phase_avg": 16})
    est._fs_hint = FS
    out = est.run(_sine())
    L   = est.latency_samples + int(0.1 * N)
    valid = out[L:]
    valid = valid[np.isfinite(valid)]
    assert len(valid) > 10
    err = abs(float(np.mean(valid)) - F0)
    assert err < 2.0, f"Mean error {err:.4f} Hz"


def test_handles_zeros():
    est = RDFTEstimator()
    est._fs_hint = FS
    out = est.run(np.zeros(N))
    assert all(np.isfinite(out))


# ─────────────────────────────────────────────────────────────────────────────
# Self-description API
# ─────────────────────────────────────────────────────────────────────────────

def test_spec_type():
    assert isinstance(RDFTEstimator.spec(), EstimatorSpec)


def test_tuning_spec_type():
    assert isinstance(RDFTEstimator.tuning_spec(), TuningSpec)


def test_tuning_spec_candidate_grid():
    ts = RDFTEstimator.tuning_spec()
    grid = ts.candidate_grid()
    assert len(grid) >= 1


def test_descriptor_keys():
    desc = RDFTEstimator.descriptor()
    for key in ("name", "family", "family_path", "complexity",
                "latency_type", "suggested_objective", "tuning_params"):
        assert key in desc


# ─────────────────────────────────────────────────────────────────────────────
# set_params / reset
# ─────────────────────────────────────────────────────────────────────────────

def test_set_params_window_size():
    est = RDFTEstimator(params={"window_size": 256})
    est._fs_hint = FS
    est.reset()
    assert est._N == 256
    est.set_params(window_size=512)
    assert est._N == 512
