"""Regression: ZeroCrossingEstimator output matches legacy pfebench reference.

Run against a fixed seed/waveform and compare RMSE to known good value.
"""

from __future__ import annotations

import numpy as np
from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz

KNOWN_RMSE_BOUND = 0.5  # Hz — conservative upper bound from legacy runs


def test_zero_crossing_parity():
    sc = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=2.0)
    sc.set_montecarlo_tuning({"seed": 42})
    out = sc.build()
    est = ZeroCrossingEstimator()
    f_hat = est.run(out.v)
    lat = est.latency_samples
    valid_hat = f_hat[lat:]
    valid_true = out.f_true[lat:]
    rmse = float(np.sqrt(np.mean((valid_hat - valid_true) ** 2)))
    assert rmse < KNOWN_RMSE_BOUND, f"RMSE {rmse:.4f} Hz exceeds bound {KNOWN_RMSE_BOUND}"
