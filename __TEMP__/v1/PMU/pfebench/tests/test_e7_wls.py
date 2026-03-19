#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/tests/test_e7_wls.py

Unit tests for E7_WLS (windowed least-squares frequency estimator).

These tests follow the same “PFEBench compatibility” pattern used in your other tests:
- set params
- reset
- run strictly online via _step()
"""

from __future__ import annotations

import numpy as np
import pytest

# ---- import estimator (try a couple common class names to avoid friction) ----
try:
    from pfebench.estimators.e7_wls import WindowedLeastSquaresEstimator as _WLS
except Exception:
    try:
        from pfebench.estimators.e7_wls import (
            RecursiveWindowedLeastSquaresEstimator as _WLS,
        )
    except Exception:
        from pfebench.estimators.e7_wls import (
            WeightedLeastSquaresWindowEstimator as _WLS,
        )  # last resort


# -----------------------------
# Helpers
# -----------------------------
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


def _assert_finite(a: np.ndarray) -> None:
    assert np.all(np.isfinite(a)), "Non-finite values found in output."


def _assert_bounded(a: np.ndarray, lo: float, hi: float) -> None:
    assert np.all(a >= lo - 1e-9), "Values below min bound."
    assert np.all(a <= hi + 1e-9), "Values above max bound."


def _set_params(est, **params):
    """
    PFEBench compatibility:
    - Some estimators implement set_params(**kwargs)
    - Others implement set_params(dict)
    - Some just expose _params directly
    """
    if hasattr(est, "set_params") and callable(getattr(est, "set_params")):
        # Try dict-style first (common in your tests)
        try:
            est.set_params(params)
        except TypeError:
            # Fall back to kwargs-style
            est.set_params(**params)
    else:
        # last fallback
        if not hasattr(est, "_params") or est._params is None:
            est._params = {}
        est._params.update(params)

    if hasattr(est, "reset") and callable(getattr(est, "reset")):
        est.reset()


def _run_online(est, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.empty_like(x, dtype=float)
    for i in range(x.size):
        # BaseEstimator-style uses _step
        y[i] = float(est._step(float(x[i])))
    return y


# -----------------------------
# Tests
# -----------------------------
def test_wls_converges_clean_60hz():
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.4)

    est = _WLS()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=51,  # window length (samples)
        search_span_hz=1.0,
        search_step_hz=0.05,
        include_dc=False,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
        freq_penalty=0.05,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    # Evaluate after warm-up (window fill)
    tail = f_hat[int(0.60 * fs) :]
    med = float(np.median(tail))
    assert abs(med - 60.0) <= 0.30


def test_wls_tracks_step_60_to_61_directionally():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    x = np.zeros_like(t)
    x[t < 0.5] = np.sin(2 * np.pi * 60.0 * t[t < 0.5])
    x[t >= 0.5] = np.sin(2 * np.pi * 61.0 * t[t >= 0.5] + 0.25)

    est = _WLS()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=61,
        search_span_hz=1.5,
        search_step_hz=0.05,
        include_dc=False,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
        freq_penalty=0.03,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    pre = float(np.mean(f_hat[int(0.30 * fs) : int(0.48 * fs)]))
    post = float(np.mean(f_hat[int(0.70 * fs) : int(0.90 * fs)]))

    # Directional tracking: should increase after +1 Hz step
    assert post > pre + 0.10
    # sanity bounds
    assert 45.0 <= pre <= 65.0
    assert 45.0 <= post <= 70.0


def test_wls_returns_finite_under_noise():
    rng = np.random.default_rng(123)
    fs = 1000.0
    t, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.3)
    x = x + 0.05 * rng.standard_normal(x.size)

    est = _WLS()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=51,
        search_span_hz=1.2,
        search_step_hz=0.05,
        include_dc=False,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
        freq_penalty=0.05,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.5 * fs) :]
    # should not collapse to min bound or explode
    frac_at_min = float(np.mean(tail <= (30.0 + 1e-9)))
    assert frac_at_min < 0.20

    med = float(np.median(tail))
    assert 45.0 <= med <= 65.0


def test_wls_handles_interharmonic_without_collapse():
    """
    Even if the estimator doesn't explicitly model interharmonics,
    it should not catastrophically collapse to min bound.
    """
    rng = np.random.default_rng(11)
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)

    f0 = 60.0
    x = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 75.0 * t + 0.4)  # interharmonic
        + 0.03 * rng.standard_normal(t.size)
    )

    est = _WLS()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=71,
        search_span_hz=1.2,
        search_step_hz=0.05,
        include_dc=False,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
        freq_penalty=0.08,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.5 * fs) :]
    frac_at_min = float(np.mean(tail <= (30.0 + 1e-9)))
    assert frac_at_min < 0.20

    med = float(np.median(tail))
    assert 40.0 <= med <= 70.0


def test_wls_handles_dc_offset_if_enabled():
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.0, amp=1.0, phase=0.1, dc=0.20)

    est = _WLS()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        win=51,
        search_span_hz=1.0,
        search_step_hz=0.05,
        include_dc=True,
        min_freq_hz=30.0,
        max_freq_hz=70.0,
        freq_penalty=0.05,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 70.0)

    tail = f_hat[int(0.6 * fs) :]
    med = float(np.median(tail))
    assert abs(med - 60.0) <= 0.40
