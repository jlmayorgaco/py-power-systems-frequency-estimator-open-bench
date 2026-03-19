# -*- coding: utf-8 -*-
"""
Unit tests for pfebench/estimators/e7_rls_basic.py (PFEBench-style BaseEstimator).
"""

import numpy as np
import pytest

from pfebench.estimators.e7_rls_basic import RecursiveLeastSquaresBasicEstimator


def _set_params(est, **params):
    """
    PFEBench compatibility:
    - Some BaseEstimator implementations use set_params(**kwargs)
    - Others store params in est._params dict
    """
    if hasattr(est, "set_params") and callable(getattr(est, "set_params")):
        # IMPORTANT: your BaseEstimator expects kwargs, not a dict positional
        est.set_params(**params)
    else:
        est._params = dict(params)
    est.reset()


def _run_online(est, x):
    y = np.empty_like(x, dtype=float)
    for k in range(len(x)):
        y[k] = float(est._step(float(x[k])))
    return y


def _sine(
    fs: float, f: float, T: float, amp: float = 1.0, phase: float = 0.0, dc: float = 0.0
):
    t = np.arange(0, T, 1.0 / fs)
    x = amp * np.sin(2 * np.pi * f * t + phase) + dc
    return t, x


def test_basic_converges_clean_60hz():
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.4, dc=0.0)

    est = RecursiveLeastSquaresBasicEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.995,
        delta=1e3,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)

    tail = f_hat[int(0.5 * fs) :]
    assert np.isfinite(tail).all()
    assert abs(float(np.mean(tail)) - 60.0) < 0.15
    assert float(np.std(tail)) < 0.40


def test_basic_tracks_step_60_to_61_relaxed():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    x = np.zeros_like(t)
    x[t < 0.5] = np.sin(2 * np.pi * 60.0 * t[t < 0.5])
    x[t >= 0.5] = np.sin(2 * np.pi * 61.0 * t[t >= 0.5] + 0.2)

    est = RecursiveLeastSquaresBasicEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.99,
        delta=1e3,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)

    pre = float(np.mean(f_hat[int(0.30 * fs) : int(0.48 * fs)]))
    post = float(np.mean(f_hat[int(0.70 * fs) : int(0.90 * fs)]))

    assert 58.0 <= pre <= 62.0
    assert 58.0 <= post <= 63.0
    assert post > pre - 0.2


def test_basic_returns_finite_under_noise():
    rng = np.random.default_rng(123)
    fs = 1000.0
    t, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.3)
    x = x + 0.05 * rng.standard_normal(x.size)

    est = RecursiveLeastSquaresBasicEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.995,
        delta=1e3,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)

    assert np.isfinite(f_hat).all()
    assert float(np.percentile(np.abs(f_hat - 60.0), 95)) < 5.0
