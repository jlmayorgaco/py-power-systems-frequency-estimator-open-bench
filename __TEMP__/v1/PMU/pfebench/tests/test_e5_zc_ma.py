#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/tests/test_E5_ZeroCrossingMAEstimator.py

Unit tests for:
  pfebench/estimators/e5_zc_ma.py  (ZeroCrossingMAEstimator)

Covers:
- API smoke: reset/step/run
- Latency model
- Accuracy on clean 60 Hz sine
- Tracks a 60->61 Hz step reasonably
"""

import math
import numpy as np
import pytest

from pfebench.estimators.e5_zc_ma import ZeroCrossingMAEstimator


def _sine_signal(
    fs: float, f_hz: float, duration_s: float, noise: float = 0.0, seed: int = 0
):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    x = np.sin(2.0 * np.pi * f_hz * t)
    if noise > 0.0:
        x = x + rng.normal(0.0, noise, size=n)
    return t, x


def _step_freq_signal(
    fs: float,
    f0: float,
    f1: float,
    step_time_s: float,
    duration_s: float,
    noise: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    f_true = np.full(n, f0, dtype=float)
    f_true[t >= step_time_s] = f1
    phi = 2.0 * np.pi * np.cumsum(f_true) / fs
    x = np.sin(phi)
    if noise > 0.0:
        x = x + rng.normal(0.0, noise, size=n)
    return t, x, f_true


def test_e5_tuning_ranges_compact_and_valid():
    tr = ZeroCrossingMAEstimator.tuning_ranges()
    assert isinstance(tr, list)
    assert len(tr) >= 4
    # no fs in tuning grid (should be scenario-driven)
    assert all(p.name != "fs" for p in tr)


def test_e5_latency_samples_matches_ma_delay():
    est = ZeroCrossingMAEstimator({"fs": 1000.0, "filter_win": 9})
    est.reset()
    # MA group delay ~ (w-1)/2
    assert est.latency_samples == int(round((9 - 1) / 2.0))


def test_e5_smoke_step_runs_no_nan():
    fs = 1000.0
    _, x = _sine_signal(fs, 60.0, 0.25, noise=0.0, seed=1)

    est = ZeroCrossingMAEstimator(
        {"fs": fs, "filter_win": 5, "use_interp": 1, "ema_alpha": 1.0}
    )
    est.reset()

    y = np.array([est.step(float(v)) for v in x], dtype=float)
    # should be finite after some samples
    assert np.isfinite(y[-1])


def test_e5_accuracy_clean_60hz():
    fs = 1000.0
    _, x = _sine_signal(fs, 60.0, 1.0, noise=0.0, seed=2)

    est = ZeroCrossingMAEstimator(
        {
            "fs": fs,
            "filter_win": 5,
            "use_interp": 1,
            "min_abs": 0.0,
            "ema_alpha": 1.0,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
            "f_nom_hz": 60.0,
        }
    )
    est.reset()
    y = np.array([est.step(float(v)) for v in x], dtype=float)

    # ignore warmup (need at least 2 crossings, plus MA delay)
    warm = int(0.20 * fs)
    tail = y[warm:]
    tail = tail[np.isfinite(tail)]

    assert tail.size > 10
    # clean sine should be close
    assert abs(float(np.median(tail)) - 60.0) < 0.25


def test_e5_tracks_step_60_to_61_reasonably():
    fs = 1000.0
    t, x, f_true = _step_freq_signal(
        fs, 60.0, 61.0, step_time_s=0.5, duration_s=1.0, noise=0.0, seed=3
    )

    est = ZeroCrossingMAEstimator(
        {
            "fs": fs,
            "filter_win": 5,
            "use_interp": 1,
            "min_abs": 0.0,
            "ema_alpha": 0.65,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
            "f_nom_hz": 60.0,
        }
    )
    est.reset()
    y = np.array([est.step(float(v)) for v in x], dtype=float)

    # before step: median close to 60
    idx_pre = (t >= 0.20) & (t < 0.45)
    pre = y[idx_pre]
    pre = pre[np.isfinite(pre)]
    assert pre.size > 10
    assert abs(float(np.median(pre)) - 60.0) < 0.5

    # after step: give it time to update via crossings + smoothing
    idx_post = (t >= 0.70) & (t <= 0.95)
    post = y[idx_post]
    post = post[np.isfinite(post)]
    assert post.size > 10
    assert abs(float(np.median(post)) - 61.0) < 0.6


def test_e5_noise_robustness_basic():
    fs = 1000.0
    _, x = _sine_signal(fs, 60.0, 1.0, noise=0.05, seed=4)

    est = ZeroCrossingMAEstimator(
        {
            "fs": fs,
            "filter_win": 9,  # stronger smoothing
            "use_interp": 1,
            "min_abs": 1e-6,
            "ema_alpha": 0.35,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
            "f_nom_hz": 60.0,
        }
    )
    est.reset()
    y = np.array([est.step(float(v)) for v in x], dtype=float)

    warm = int(0.25 * fs)
    tail = y[warm:]
    tail = tail[np.isfinite(tail)]
    assert tail.size > 10

    # not super strict—just ensure it's not drifting wildly
    med = float(np.median(tail))
    assert 57.0 < med < 63.0
