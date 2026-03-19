#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e7_wls.py

E7 — Windowed Least Squares (WLS) Frequency Estimator (classic sinusoid-fit, sliding window).

Idea
----
Fit a single-tone sinusoid on a sliding window (strictly causal) using linear least-squares for
each candidate frequency:

    x[n] ≈ A cos(ω nTs) + B sin(ω nTs) + C   (optional DC)

For a *fixed* ω, this is linear in [A,B,C] so we can solve LS on a window of W samples.

Frequency is nonlinear, so we do a small 1D search around previous f_hat on each step:
  - build a candidate grid
  - for each f: solve LS on current window -> residual SSE(f)
  - pick best f (plus optional smoothness/nominal penalties)
  - output f_hat, update state, slide window

This is the "batch LS over a window" version of what you described (MLE under i.i.d. Gaussian noise).

PFEBench integration
--------------------
- Inherits BaseEstimator
- Implements tuning_ranges(), reset(), latency_samples, _step()
- Strictly online: uses only past+current samples (causal window).

Notes
-----
- Complexity: O(|grid| * W) per sample. With small W (e.g., 25..101) and small grids it is fine.
- This is a good baseline: smoother than AR(2)-RLS, and more interpretable.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class WindowedLeastSquaresEstimator(BaseEstimator):
    NAME: str = "WindowedLeastSquaresEstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Keep it compact. fs comes from scenario, do NOT tune it.
        """
        return [
            TuningParam(
                name="wls_win",
                default=51,
                type="int",
                values=[25, 51, 101],
                description="Sliding window length W (samples). Larger = smoother, slower.",
            ),
            TuningParam(
                name="include_dc",
                default=True,
                type="bool",
                values=[False, True],
                description="Include DC term C in the sinusoid fit.",
            ),
            TuningParam(
                name="search_span_hz",
                default=1.0,
                type="float",
                values=[0.5, 1.0, 1.5],
                description="Local +/- frequency search span (Hz).",
            ),
            TuningParam(
                name="search_step_hz",
                default=0.05,
                type="float",
                values=[0.02, 0.05, 0.10],
                description="Frequency grid step (Hz).",
            ),
            TuningParam(
                name="freq_penalty",
                default=0.02,
                type="float",
                values=[0.0, 0.02, 0.05, 0.10],
                description="Penalty to discourage big per-step frequency changes.",
            ),
            TuningParam(
                name="nom_penalty",
                default=0.0,
                type="float",
                values=[0.0, 0.001, 0.002],
                description="Weak prior to nominal frequency to avoid drift.",
            ),
            TuningParam(
                name="min_freq_hz",
                default=30.0,
                type="float",
                values=[20.0, 30.0, 40.0],
                description="Minimum plausible frequency clamp (Hz).",
            ),
            TuningParam(
                name="max_freq_hz",
                default=70.0,
                type="float",
                values=[70.0, 80.0, 90.0],
                description="Maximum plausible frequency clamp (Hz).",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._W = int(self._params.get("wls_win", 51))
        self._include_dc = bool(self._params.get("include_dc", True))

        self._search_span = float(self._params.get("search_span_hz", 1.0))
        self._search_step = float(self._params.get("search_step_hz", 0.05))

        self._freq_penalty = float(self._params.get("freq_penalty", 0.02))
        self._nom_penalty = float(self._params.get("nom_penalty", 0.0))

        self._min_f = float(self._params.get("min_freq_hz", 30.0))
        self._max_f = float(self._params.get("max_freq_hz", 70.0))

        if self._W < 5:
            self._W = 5
        if self._search_step <= 0:
            self._search_step = 0.05
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0
        if self._freq_penalty < 0:
            self._freq_penalty = 0.0
        if self._nom_penalty < 0:
            self._nom_penalty = 0.0

        # sample buffer and time index
        self._buf = deque(maxlen=self._W)  # oldest -> newest
        self._k = -1

        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

    @property
    def latency_samples(self) -> int:
        # causal window: output corresponds roughly to the window "end"
        # In practice, we report 0 because we're producing per-sample online estimates.
        return 0

    # -----------------------
    # Helpers
    # -----------------------
    def _candidate_grid(self, f_prev: float) -> np.ndarray:
        span = float(self._search_span)
        step = float(self._search_step)

        if span <= 0.0:
            return np.array([f_prev], dtype=float)

        n_side = int(np.floor(span / step))
        grid = f_prev + step * np.arange(-n_side, n_side + 1, dtype=float)
        grid = np.clip(grid, self._min_f, self._max_f)
        return np.unique(grid)

    def _solve_ls_sse(self, y: np.ndarray, k0: int, f_hz: float) -> float:
        """
        Build design matrix on-the-fly for window:
          y[i] ~ A cos(w*(k0+i)) + B sin(w*(k0+i)) + (C if include_dc)
        Return SSE = ||y - X beta||^2
        """
        W = y.size
        idx = k0 + np.arange(W, dtype=float)  # absolute sample indices
        w = 2.0 * np.pi * float(f_hz) / self._fs

        c = np.cos(w * idx)
        s = np.sin(w * idx)

        if self._include_dc:
            X = np.column_stack([c, s, np.ones(W, dtype=float)])
        else:
            X = np.column_stack([c, s])

        # Solve LS (robust)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        e = y - y_hat
        return float(e.T @ e)

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        xk = float(v_sample)
        if not np.isfinite(xk):
            return float(self._f_est)

        self._k += 1
        self._buf.append(xk)

        # Not enough samples yet -> keep nominal estimate
        if len(self._buf) < self._W:
            return float(self._f_est)

        y = np.asarray(self._buf, dtype=float)  # length W
        # absolute index of first sample in the window
        k0 = int(self._k - (self._W - 1))

        f_prev = float(self._f_est)
        grid = self._candidate_grid(f_prev)

        best_f = f_prev
        best_cost = float("inf")

        for f_c in grid:
            sse = self._solve_ls_sse(y=y, k0=k0, f_hz=float(f_c))
            cost = float(sse)

            if self._freq_penalty > 0.0:
                df = float(f_c - f_prev)
                cost += float(self._freq_penalty * (df * df))

            if self._nom_penalty > 0.0:
                dn = float(f_c - self._f_nom)
                cost += float(self._nom_penalty * (dn * dn))

            if cost < best_cost:
                best_cost = cost
                best_f = float(f_c)

        self._f_est = float(_clamp(best_f, self._min_f, self._max_f))
        return float(self._f_est)
