"""Unit tests for ZeroCrossingEstimator."""

from __future__ import annotations

import numpy as np
from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator


def test_reset_clears_state():
    est = ZeroCrossingEstimator()
    est.reset()
    assert est.latency_samples >= 0


def test_run_pure_60hz_returns_near_60():
    est = ZeroCrossingEstimator()
    fs = 10_000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    v = np.sin(2 * np.pi * 60.0 * t)
    f_hat = est.run(v)
    # Trim latency
    lat = est.latency_samples
    valid = f_hat[lat:]
    assert np.abs(np.nanmean(valid) - 60.0) < 0.5, (
        f"Mean freq {np.nanmean(valid):.3f} not near 60 Hz"
    )


def test_set_params_then_reset():
    est = ZeroCrossingEstimator()
    est.set_params(filter_win=7)
    est.reset()
