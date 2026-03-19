#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/tests/test_e10_nr.py

Unit tests for pfebench.estimators.e10_nr.NewtonRaphsonFrequencyEstimator
"""

import numpy as np

from pfebench.estimators.e10_nr import NewtonRaphsonFrequencyEstimator


def _set_params(est, **params):
    if hasattr(est, "set_params") and callable(getattr(est, "set_params")):
        est.set_params(**params)
    if hasattr(est, "reset"):
        est.reset()


def _run_online(est, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        out[i] = float(est.step(float(x[i])))
    return out


def _assert_finite(arr: np.ndarray):
    assert np.all(np.isfinite(arr))


def _assert_bounded(arr: np.ndarray, lo: float, hi: float):
    assert np.min(arr) >= lo - 1e-9
    assert np.max(arr) <= hi + 1e-9


def _sine(
    fs: float,
    f: float,
    T: float = 1.0,
    amp: float = 1.0,
    phase: float = 0.0,
    dc: float = 0.0,
):
    t = np.arange(0, T, 1.0 / fs)
    x = dc + amp * np.sin(2.0 * np.pi * f * t + phase)
    return t, x


def test_nr_converges_clean_60hz():
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.4, dc=0.0)

    est = NewtonRaphsonFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=201,
        nr_steps=2,
        fd_hz=0.05,
        ridge=1e-6,
        ema_alpha=0.6,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.7 * fs) :]
    med = float(np.median(tail))
    assert 58.5 <= med <= 61.5


def test_nr_returns_finite_under_noise():
    rng = np.random.default_rng(123)
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.3, dc=0.0)
    x = x + 0.05 * rng.standard_normal(x.size)

    est = NewtonRaphsonFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=201,
        nr_steps=2,
        fd_hz=0.05,
        ridge=1e-6,
        ema_alpha=0.6,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.7 * fs) :]
    med = float(np.median(tail))
    assert 55.0 <= med <= 65.0


def test_nr_tracks_step_60_to_61_sane():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    x = np.zeros_like(t)
    x[t < 0.5] = np.sin(2 * np.pi * 60.0 * t[t < 0.5])
    x[t >= 0.5] = np.sin(2 * np.pi * 61.0 * t[t >= 0.5] + 0.3)

    est = NewtonRaphsonFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=151,  # shorter => better step response
        nr_steps=2,
        fd_hz=0.05,
        ridge=1e-6,
        ema_alpha=0.6,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    pre = float(np.mean(f_hat[int(0.30 * fs) : int(0.48 * fs)]))
    post = float(np.mean(f_hat[int(0.70 * fs) : int(0.90 * fs)]))

    assert 50.0 <= pre <= 65.0
    assert 50.0 <= post <= 70.0
    assert post >= pre - 0.8


def test_nr_handles_interharmonic_without_collapse():
    rng = np.random.default_rng(11)
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    f0 = 60.0
    x = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 75.0 * t + 0.4)  # interharmonic
        + 0.03 * rng.standard_normal(t.size)
    )

    est = NewtonRaphsonFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=201,
        nr_steps=2,
        fd_hz=0.05,
        ridge=1e-6,
        ema_alpha=0.6,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.7 * fs) :]
    frac_at_min = float(np.mean(tail <= (30.0 + 1e-9)))
    assert frac_at_min < 0.20

    med = float(np.median(tail))
    assert 40.0 <= med <= 70.0
