#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e6_mwls.py

E6 — Moving-Window Least Squares (MWLS) Frequency Estimator (Q1-grade baseline).

Idea (phase-slope LS)
---------------------
1) Build analytic signal z[n] = I[n] + j Q[n] using a causal FIR Hilbert transformer.
2) Compute phase: φ[n] = atan2(Q[n], I[n]).
3) Over a sliding window of W samples, unwrap phase incrementally and fit a line:

      φ[k] ≈ m*k + b   (k in samples within the window)

   Closed-form LS slope:
      m = cov(k, φ) / var(k)

4) Convert slope to frequency:
      f = (fs / (2π)) * m

Notes
-----
- This is a robust, transparent regression baseline: "MWLS on phase".
- It naturally smooths noise via regression over a window.
- Includes optional input MA, magnitude gate, min/max bounds, and EMA on output.

Status: Early development – API not stable.
"""

from __future__ import annotations

from collections import deque
from typing import List

import math
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class MovingWindowLeastSquaresEstimator(BaseEstimator):
    """
    MWLS frequency estimator using LS slope of unwrapped phase over a moving window.
    """

    NAME: str = "MovingWindowLeastSquaresEstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        IMPORTANT:
        - Do NOT tune `fs` (comes from scenario / runner config).
        - Do NOT tune `f_nom_hz` (only initialization / latency approximation).
        Keep the grid compact.
        """
        return [
            TuningParam(
                name="filter_win",
                default=1,
                type="int",
                values=[1, 3, 5, 9],
                description="Input moving-average pre-filter window. 1 disables.",
            ),
            TuningParam(
                name="hilbert_len",
                default=51,
                type="int",
                values=[31, 51, 81],
                description="Odd length of FIR Hilbert transformer.",
            ),
            TuningParam(
                name="mwls_win",
                default=25,
                type="int",
                values=[11, 21, 31, 41, 51],
                description="Moving window length W (samples) for LS phase-slope. Must be >= 3.",
            ),
            TuningParam(
                name="min_abs",
                default=1e-6,
                type="float",
                values=[0.0, 1e-6, 1e-4],
                description="Minimum analytic magnitude |z| gate to accept phase samples.",
            ),
            TuningParam(
                name="ema_alpha",
                default=0.65,
                type="float",
                values=[1.0, 0.65, 0.35],
                description="EMA smoothing on frequency. 1.0 = no smoothing.",
            ),
            TuningParam(
                name="min_freq_hz",
                default=40.0,
                type="float",
                values=[30.0, 40.0],
                description="Minimum plausible frequency (Hz) for outlier rejection.",
            ),
            TuningParam(
                name="max_freq_hz",
                default=80.0,
                type="float",
                values=[70.0, 80.0, 90.0],
                description="Maximum plausible frequency (Hz) for outlier rejection.",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        # fs comes from runner/scenario; keep but do not tune
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._filter_win = max(1, int(self._params.get("filter_win", 1)))
        self._hilbert_len = int(self._params.get("hilbert_len", 51))
        self._mwls_win = int(self._params.get("mwls_win", 25))

        self._min_abs = float(self._params.get("min_abs", 1e-6))
        self._ema_alpha = float(self._params.get("ema_alpha", 0.65))
        self._min_f = float(self._params.get("min_freq_hz", 40.0))
        self._max_f = float(self._params.get("max_freq_hz", 80.0))

        # enforce odd hilbert_len >= 3
        if self._hilbert_len < 3:
            self._hilbert_len = 3
        if self._hilbert_len % 2 == 0:
            self._hilbert_len += 1

        # enforce mwls_win >= 3
        if self._mwls_win < 3:
            self._mwls_win = 3

        # clamp EMA alpha
        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 0.65

        # Buffers (oldest -> newest)
        self._xbuf = deque(maxlen=self._filter_win)
        self._hbuf = deque(maxlen=self._hilbert_len)
        self._ibuf = deque(maxlen=self._hilbert_len)

        # Phase window buffers
        self._phi_win = deque(maxlen=self._mwls_win)  # unwrapped phases
        self._k_win = deque(maxlen=self._mwls_win)  # sample indices (integers)

        # Hilbert FIR
        self._h = self._design_hilbert_fir(self._hilbert_len)
        self._M = (self._hilbert_len - 1) // 2

        # State
        self._sample_idx = -1
        self._phi_prev_raw = None
        self._phi_prev_unwrapped = None
        self._f_est = float(self._f_nom)

    @property
    def latency_samples(self) -> int:
        # MA delay + Hilbert group delay + centered window delay
        d_in = (self._filter_win - 1) / 2.0
        d_h = (self._hilbert_len - 1) / 2.0
        d_w = (self._mwls_win - 1) / 2.0
        return int(round(d_in + d_h + d_w))

    # -----------------------
    # DSP helpers
    # -----------------------
    def _ma_in(self, x: float) -> float:
        self._xbuf.append(float(x))
        if self._filter_win == 1:
            return float(x)
        return float(sum(self._xbuf) / len(self._xbuf))

    @staticmethod
    def _wrap_to_pi(dphi: float) -> float:
        return float(np.arctan2(np.sin(dphi), np.cos(dphi)))

    @staticmethod
    def _design_hilbert_fir(L: int) -> np.ndarray:
        """
        Windowed FIR Hilbert transformer (odd length).
        Ideal impulse response:
          h[n] = 2/(π n) for n odd, 0 for n even, with n = k - (L-1)/2
        Apply Hann window.
        """
        M = (L - 1) // 2
        h = np.zeros(L, dtype=float)
        for k in range(L):
            n = k - M
            if n == 0 or (n % 2 == 0):
                h[k] = 0.0
            else:
                h[k] = 2.0 / (np.pi * n)

        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / (L - 1))
        h *= w

        # zero-DC numerical tweak (safe)
        h -= np.mean(h)
        return h

    def _hilbert_quadrature(self, x: float) -> float:
        """
        Causal FIR:
          q[n] = Σ_{k=0}^{L-1} h[k] * x[n-k]
        With deque oldest->newest, x[n-k] is at index L-1-k => xb[::-1].
        """
        self._hbuf.append(float(x))
        if len(self._hbuf) < self._hilbert_len:
            return 0.0
        xb = np.array(self._hbuf, dtype=float)  # oldest..newest
        return float(np.dot(self._h, xb[::-1]))

    def _update_unwrapped_phase(self, phi_raw: float) -> float:
        """
        Incremental unwrap:
          phi_u[n] = phi_u[n-1] + wrap(phi_raw[n] - phi_raw[n-1])
        """
        if self._phi_prev_raw is None or self._phi_prev_unwrapped is None:
            self._phi_prev_raw = float(phi_raw)
            self._phi_prev_unwrapped = float(phi_raw)
            return float(phi_raw)

        dphi = self._wrap_to_pi(float(phi_raw) - float(self._phi_prev_raw))
        phi_u = float(self._phi_prev_unwrapped + dphi)

        self._phi_prev_raw = float(phi_raw)
        self._phi_prev_unwrapped = float(phi_u)
        return float(phi_u)

    @staticmethod
    def _ls_slope(k: np.ndarray, y: np.ndarray) -> float:
        """
        slope m = cov(k,y)/var(k) (stable, closed-form).
        """
        if k.size < 3:
            return float("nan")
        k_mean = float(np.mean(k))
        y_mean = float(np.mean(y))
        dk = k - k_mean
        dy = y - y_mean
        denom = float(np.sum(dk * dk))
        if abs(denom) < 1e-18:
            return float("nan")
        return float(np.sum(dk * dy) / denom)

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        if not math.isfinite(v_sample):
            return float(self._f_est)

        self._sample_idx += 1

        # 1) Optional input MA
        x = self._ma_in(float(v_sample))

        # 2) Hilbert quadrature (delayed by M)
        q = self._hilbert_quadrature(x)

        # 3) Delay line for I alignment
        self._ibuf.append(float(x))
        if len(self._ibuf) < self._hilbert_len:
            return float(self._f_est)

        I = float(self._ibuf[self._M])  # aligns with q
        Q = float(q)

        # 4) Magnitude gate
        mag = math.hypot(I, Q)
        if mag < self._min_abs:
            return float(self._f_est)

        # 5) Phase (raw) then unwrap incrementally
        phi_raw = float(np.arctan2(Q, I))
        phi_u = self._update_unwrapped_phase(phi_raw)

        # 6) Update window buffers
        self._phi_win.append(phi_u)
        self._k_win.append(int(self._sample_idx))

        if len(self._phi_win) < max(3, self._mwls_win):
            return float(self._f_est)

        # 7) LS slope over window
        k = np.asarray(self._k_win, dtype=float)
        y = np.asarray(self._phi_win, dtype=float)

        m = self._ls_slope(k, y)  # rad/sample
        if (not math.isfinite(m)) or abs(m) < self.EPS:
            return float(self._f_est)

        f_new = float((self._fs / (2.0 * np.pi)) * m)

        # robust sign convention
        if math.isfinite(f_new) and (f_new < 0.0):
            f_new = -f_new

        # 8) Bounds / outlier rejection
        if (
            (not math.isfinite(f_new))
            or (f_new <= self._min_f)
            or (f_new >= self._max_f)
        ):
            return float(self._f_est)

        # 9) EMA smoothing
        a = float(self._ema_alpha)
        if a <= 0.0:
            return float(self._f_est)
        if a >= 1.0:
            self._f_est = float(f_new)
        else:
            self._f_est = float((1.0 - a) * self._f_est + a * float(f_new))

        return float(self._f_est)
