"""Unit tests for compute_metrics."""
from __future__ import annotations

import numpy as np
import pytest
from openfreqbench.metrics.frequency import compute_metrics, MetricConfig


def test_compute_metrics_perfect_estimate():
    fs = 10_000.0
    n = int(fs * 1.0)
    f_true = np.full(n, 60.0)
    f_hat = np.full(n, 60.0)
    cfg = MetricConfig(fs_hz=fs, f_nom=60.0)
    metrics = compute_metrics(f_hat, f_true, exec_time_s=0.001, latency_samples=0, cfg=cfg)
    assert metrics["RMSE_HZ"]["value"] < 1e-10


def test_compute_metrics_returns_expected_keys():
    fs = 1_000.0
    n = 1000
    f_true = np.full(n, 60.0)
    f_hat = np.full(n, 60.1)
    cfg = MetricConfig(fs_hz=fs, f_nom=60.0)
    metrics = compute_metrics(f_hat, f_true, exec_time_s=0.001, latency_samples=0, cfg=cfg)
    for key in ["RMSE_HZ", "MAE_HZ", "BIAS_HZ", "TIME_PER_SAMPLE_US", "LATENCY_SAMPLES"]:
        assert key in metrics, f"Missing key: {key}"
