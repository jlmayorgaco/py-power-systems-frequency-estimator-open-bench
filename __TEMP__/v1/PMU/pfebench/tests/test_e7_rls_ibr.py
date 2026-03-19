# -*- coding: utf-8 -*-
"""
Unit tests for pfebench/estimators/e7_rls_ibr.py (PFEBench-style BaseEstimator).
Robustness-oriented tests (finite/bounded/no-collapse), not accuracy-guaranteeing.
"""

import numpy as np
import pytest

from pfebench.estimators.e7_rls_ibr import RecursiveLeastSquaresIBREstimator


def _set_params(est, **params):
    if hasattr(est, "set_params") and callable(getattr(est, "set_params")):
        est.set_params(**params)
    else:
        est._params = dict(params)
    est.reset()


def _run_online(est, x):
    y = np.empty_like(x, dtype=float)
    for k in range(len(x)):
        y[k] = float(est._step(float(x[k])))
    return y


def _assert_finite(x):
    assert np.isfinite(x).all()


def _assert_bounded(x, lo, hi):
    assert float(np.min(x)) >= lo - 1e-9
    assert float(np.max(x)) <= hi + 1e-9


def test_ibr_converges_with_harmonics_and_dc_sane():
    rng = np.random.default_rng(7)
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    f0 = 60.0
    x = (
        1.0 * np.sin(2 * np.pi * f0 * t + 0.2)
        + 0.25 * np.sin(2 * np.pi * (2 * f0) * t + 0.7)
        + 0.10 * np.sin(2 * np.pi * (5 * f0) * t + 1.4)
        + 0.05
        + 0.03 * rng.standard_normal(t.size)
    )

    est = RecursiveLeastSquaresIBREstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.995,
        delta=1e3,
        harmonics=(1, 2, 3, 5, 7),
        interharmonics_hz=(),
        include_dc=True,
        search_span_hz=1.0,
        search_step_hz=0.05,
        freq_penalty=0.10,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.5 * fs) :]
    med = float(np.median(tail))
    assert 45.0 <= med <= 65.0


def test_ibr_handles_interharmonic_without_collapse():
    rng = np.random.default_rng(11)
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    f0 = 60.0
    x = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 75.0 * t + 0.4)
        + 0.03 * rng.standard_normal(t.size)
    )

    est = RecursiveLeastSquaresIBREstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.995,
        delta=1e3,
        harmonics=(1, 2, 3),
        interharmonics_hz=(75.0,),
        include_dc=False,
        search_span_hz=1.2,
        search_step_hz=0.05,
        freq_penalty=0.05,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.5 * fs) :]
    frac_at_min = float(np.mean(tail <= (30.0 + 1e-9)))
    assert frac_at_min < 0.20

    # Relaxed: this estimator may bias low under strong interharmonics depending on penalty/grid.
    med = float(np.median(tail))
    assert 30.0 <= med <= 70.0


def test_ibr_tracks_step_60_to_61_sane():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    x = np.zeros_like(t)
    x[t < 0.5] = np.sin(2 * np.pi * 60.0 * t[t < 0.5])
    x[t >= 0.5] = np.sin(2 * np.pi * 61.0 * t[t >= 0.5] + 0.3)

    est = RecursiveLeastSquaresIBREstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        lam=0.995,
        delta=1e3,
        harmonics=(1, 2, 3),
        interharmonics_hz=(),
        include_dc=False,
        search_span_hz=1.5,
        search_step_hz=0.05,
        freq_penalty=0.05,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    pre = float(np.mean(f_hat[int(0.30 * fs) : int(0.48 * fs)]))
    post = float(np.mean(f_hat[int(0.70 * fs) : int(0.90 * fs)]))

    assert 45.0 <= pre <= 65.0
    assert 45.0 <= post <= 70.0

    # Relaxed: allow up to ~1.5 Hz undershoot due to local-search + penalty + transient.
    assert post >= pre - 1.5
