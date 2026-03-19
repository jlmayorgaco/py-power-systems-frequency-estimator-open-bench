#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/tests/test_e6_mwls.py

Unit tests for:
- pfebench/estimators/e6_mwls.py
  MovingWindowLeastSquaresEstimator (E6_MWLS)

Goals (Q1-grade sanity):
- deterministic constant-frequency tracking (low noise)
- reasonable response to a step in frequency
- robustness to NaN/Inf inputs (hold last estimate)
- latency_samples is finite and non-negative
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pfebench.estimators.e6_mwls import MovingWindowLeastSquaresEstimator


def _synth_sine(
    fs: float, f_hz: float, duration_s: float, noise_std: float, seed: int = 0
):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    phi = 2.0 * np.pi * f_hz * t
    x = np.sin(phi)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=n)
    return t, x


def _synth_step_freq(
    fs: float,
    f0: float,
    f1: float,
    step_time_s: float,
    duration_s: float,
    noise_std: float,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    f_true = np.full(n, f0, dtype=float)
    f_true[t >= step_time_s] = f1
    # integrate frequency to phase
    phi = 2.0 * np.pi * np.cumsum(f_true) / fs
    x = np.sin(phi)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=n)
    return t, x, f_true


def _run_stream(est, x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    est.reset()
    for i, xi in enumerate(x):
        y[i] = float(est.step(float(xi)))
    return y


def test_e6_mwls_latency_nonnegative():
    est = MovingWindowLeastSquaresEstimator(
        {
            "fs": 1000.0,
            "filter_win": 3,
            "hilbert_len": 51,
            "mwls_win": 31,
            "ema_alpha": 1.0,
        }
    )
    est.reset()
    lat = int(getattr(est, "latency_samples", -1))
    assert lat >= 0
    assert math.isfinite(float(lat))


def test_e6_mwls_constant_frequency_low_noise_tracks_reasonably():
    fs = 1000.0
    f0 = 60.0
    _, x = _synth_sine(fs=fs, f_hz=f0, duration_s=2.0, noise_std=0.001, seed=1)

    est = MovingWindowLeastSquaresEstimator(
        {
            "fs": fs,
            "filter_win": 3,
            "hilbert_len": 51,
            "mwls_win": 31,
            "min_abs": 1e-8,
            "ema_alpha": 1.0,  # no smoothing (test raw behavior)
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )
    y = _run_stream(est, x)

    # ignore warm-up (latency + some margin)
    warm = max(200, int(est.latency_samples) + 50)
    tail = y[warm:]
    assert tail.size > 100

    # Robust summary: median absolute error
    med_abs_err = float(np.median(np.abs(tail - f0)))
    # MWLS baseline should be close at low noise; keep tolerance realistic for FIR Hilbert + LS
    assert med_abs_err < 0.25


def test_e6_mwls_step_frequency_converges_after_delay():
    fs = 1000.0
    f0, f1 = 60.0, 61.0
    step_time_s = 0.5
    duration_s = 1.5

    t, x, f_true = _synth_step_freq(
        fs=fs,
        f0=f0,
        f1=f1,
        step_time_s=step_time_s,
        duration_s=duration_s,
        noise_std=0.002,
        seed=2,
    )

    est = MovingWindowLeastSquaresEstimator(
        {
            "fs": fs,
            "filter_win": 3,
            "hilbert_len": 51,
            "mwls_win": 41,
            "min_abs": 1e-8,
            "ema_alpha": 0.65,  # a bit of smoothing
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )
    y = _run_stream(est, x)

    # After the step + latency + settling margin, estimator should be closer to f1 than f0
    lat = int(est.latency_samples)
    idx_step = int(round(step_time_s * fs))
    idx_check = min(len(y) - 1, idx_step + lat + 250)

    # use a window near the end for stability
    tail = y[idx_check:]
    assert tail.size > 50

    mean_tail = float(np.mean(tail))
    assert abs(mean_tail - f1) < abs(mean_tail - f0)
    assert abs(mean_tail - f1) < 0.5


def test_e6_mwls_nan_inf_hold_last_value():
    fs = 1000.0
    f0 = 60.0
    _, x = _synth_sine(fs=fs, f_hz=f0, duration_s=1.0, noise_std=0.0, seed=3)

    est = MovingWindowLeastSquaresEstimator(
        {
            "fs": fs,
            "filter_win": 1,
            "hilbert_len": 31,
            "mwls_win": 21,
            "min_abs": 1e-8,
            "ema_alpha": 1.0,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )
    est.reset()

    # warm-up with some samples
    last = None
    for i in range(200):
        last = float(est.step(float(x[i])))

    assert last is not None and math.isfinite(last)

    # feed NaN/Inf and ensure it holds
    y_nan = float(est.step(float("nan")))
    y_inf = float(est.step(float("inf")))
    assert y_nan == pytest.approx(last)
    assert y_inf == pytest.approx(last)
