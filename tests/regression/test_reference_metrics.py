"""Regression: compute_metrics returns stable values for known input."""
from __future__ import annotations

import numpy as np
from openfreqbench.metrics.frequency import compute_metrics, MetricConfig


def test_reference_metrics_stable():
    fs = 10_000.0
    n = int(fs * 1.0)
    f_true = np.full(n, 60.0)
    f_hat = f_true + 0.1  # constant 0.1 Hz offset
    cfg = MetricConfig(fs_hz=fs, f_nom=60.0)
    m = compute_metrics(f_hat, f_true, exec_time_s=0.001, latency_samples=0, cfg=cfg)
    assert abs(m["RMSE_HZ"]["value"] - 0.1) < 1e-6
    assert abs(m["MAE_HZ"]["value"] - 0.1) < 1e-6
    assert abs(m["BIAS_HZ"]["value"] - 0.1) < 1e-6
