#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/tests/test_e8_ar.py

Unit tests for:
  pfebench.estimators.e8_ar.AutoregressiveFrequencyEstimator

Style: mirrors E7 tests (online run, set_params compatibility, sanity checks).
"""

from __future__ import annotations

import numpy as np
import pytest

from pfebench.estimators.e8_ar import AutoregressiveFrequencyEstimator


# -----------------------------
# Helpers (PFEBench-style)
# -----------------------------
def _set_params(est, **params):
    """
    PFEBench compatibility:
    - some estimators accept set_params(**kwargs)
    - PFEBench BaseEstimator typically uses set_params(**kwargs)
    """
    if hasattr(est, "set_params") and callable(getattr(est, "set_params")):
        try:
            est.set_params(**params)
        except TypeError:
            # fallback if someone's set_params expects a dict
            est.set_params(params)
    else:
        # last resort: direct param injection (not preferred)
        if hasattr(est, "_params") and isinstance(getattr(est, "_params"), dict):
            est._params.update(params)
    if hasattr(est, "reset") and callable(getattr(est, "reset")):
        est.reset()


def _run_online(est, x: np.ndarray) -> np.ndarray:
    """
    Run estimator strictly online, calling _step if present (BaseEstimator pattern),
    otherwise step() if implemented.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.empty_like(x, dtype=float)

    if hasattr(est, "_step") and callable(getattr(est, "_step")):
        for k in range(x.size):
            y[k] = float(est._step(float(x[k])))
        return y

    if hasattr(est, "step") and callable(getattr(est, "step")):
        for k in range(x.size):
            y[k] = float(est.step(float(x[k])))
        return y

    # Fallback: estimate vectorized (still acceptable if it's internally online)
    if hasattr(est, "estimate") and callable(getattr(est, "estimate")):
        return np.asarray(
            est.estimate(
                x, fs=getattr(est, "_fs", 1000.0), f_nom_hz=getattr(est, "_f_nom", 60.0)
            )
        )

    raise RuntimeError("Estimator has no _step/step/estimate API.")


def _assert_finite(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    assert np.all(np.isfinite(arr))


def _assert_bounded(arr: np.ndarray, lo: float, hi: float):
    arr = np.asarray(arr, dtype=float)
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
    t = np.arange(0.0, T, 1.0 / fs)
    x = dc + amp * np.sin(2.0 * np.pi * f * t + phase)
    return t, x


# -----------------------------
# Tests
# -----------------------------
def test_e8_ar_converges_clean_60hz():
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.2, amp=1.0, phase=0.4, dc=0.0)

    est = AutoregressiveFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        ar_order=6,
        win=201,
        ema_alpha=0.60,
        min_freq_hz=40.0,
        max_freq_hz=80.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 40.0, 80.0)

    # After window is filled + some margin, should be close to 60 Hz
    start = int(0.8 * fs)
    mean_tail = float(np.mean(f_hat[start:]))
    assert abs(mean_tail - 60.0) <= 0.30


def test_e8_ar_returns_finite_under_noise():
    rng = np.random.default_rng(123)
    fs = 1000.0
    _, x = _sine(fs, 60.0, T=1.2, amp=1.0, phase=0.3, dc=0.0)
    x = x + 0.05 * rng.standard_normal(x.size)

    est = AutoregressiveFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        ar_order=8,
        win=301,
        ema_alpha=0.60,
        min_freq_hz=30.0,
        max_freq_hz=90.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 90.0)

    tail = f_hat[int(0.8 * fs) :]
    med = float(np.median(tail))
    # loose sanity: should not be crazy
    assert 55.0 <= med <= 65.0


def test_e8_ar_tracks_step_60_to_61_directionally():
    fs = 1000.0
    t = np.arange(0, 1.5, 1.0 / fs)

    x = np.zeros_like(t)
    x[t < 0.75] = np.sin(2 * np.pi * 60.0 * t[t < 0.75])
    x[t >= 0.75] = np.sin(2 * np.pi * 61.0 * t[t >= 0.75] + 0.2)

    est = AutoregressiveFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        ar_order=6,
        win=201,
        ema_alpha=0.60,
        min_freq_hz=40.0,
        max_freq_hz=80.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 40.0, 80.0)

    # windows: ensure we're well after fill and away from transition edges
    pre = float(np.mean(f_hat[int(0.45 * fs) : int(0.70 * fs)]))
    post = float(np.mean(f_hat[int(1.10 * fs) : int(1.35 * fs)]))

    # should move upward at least a bit
    assert post > pre + 0.10
    # and stay in plausible band
    assert 55.0 <= pre <= 65.0
    assert 55.0 <= post <= 70.0


def test_e8_ar_handles_dc_offset_without_collapse():
    rng = np.random.default_rng(7)
    fs = 1000.0
    t = np.arange(0, 1.2, 1.0 / fs)

    x = (
        1.0 * np.sin(2 * np.pi * 60.0 * t + 0.2)
        + 0.10  # DC
        + 0.03 * rng.standard_normal(t.size)
    )

    est = AutoregressiveFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        ar_order=6,
        win=201,
        ema_alpha=0.60,
        min_freq_hz=30.0,
        max_freq_hz=90.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 90.0)

    tail = f_hat[int(0.8 * fs) :]
    med = float(np.median(tail))
    assert 55.0 <= med <= 65.0


def test_e8_ar_interharmonic_not_insane_on_median():
    """
    With an interharmonic present, AR can drift, but should not collapse to silly values.
    This is a sanity test, not a strict "must equal 60".
    """
    rng = np.random.default_rng(11)
    fs = 1000.0
    t = np.arange(0, 1.2, 1.0 / fs)

    x = (
        1.0 * np.sin(2 * np.pi * 60.0 * t)
        + 0.15 * np.sin(2 * np.pi * 75.0 * t + 0.4)  # interharmonic
        + 0.03 * rng.standard_normal(t.size)
    )

    est = AutoregressiveFrequencyEstimator()
    _set_params(
        est,
        fs=fs,
        f_nom_hz=60.0,
        ar_order=10,
        win=301,
        ema_alpha=0.60,
        min_freq_hz=30.0,
        max_freq_hz=90.0,
    )

    f_hat = _run_online(est, x)
    _assert_finite(f_hat)
    _assert_bounded(f_hat, 30.0, 90.0)

    tail = f_hat[int(0.8 * fs) :]
    med = float(np.median(tail))

    # Should not collapse to low bound / nonsense.
    assert 40.0 <= med <= 90.0

    # Not stuck at min bound too often
    frac_at_min = float(np.mean(tail <= (30.0 + 1e-9)))
    assert frac_at_min < 0.20
