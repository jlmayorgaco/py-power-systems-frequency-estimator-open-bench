#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e10_nr.py

E10 — Newton-Raphson Frequency Estimator (windowed, time-domain, PFEBench-style).

Core idea
---------
Estimate frequency f by minimizing the LS residual of fitting a single-tone sinusoid:

    x[n] ≈ A cos(w t_n) + B sin(w t_n) + C

For a candidate w = 2π f:
- The parameters [A,B,C] are linear -> solve by (regularized) LS on the window.
- The residual SSE(w) = ||x - Phi(w) theta(w)||^2 is scalar in w.
- We update w via Newton-Raphson:
      w <- w - g(w)/H(w)
  where g,H are numerical derivatives of SSE(w) w.r.t w (finite differences).
This is robust and simple, avoids symbolic derivatives.

Strictly-online behavior
------------------------
At each sample:
- update ring buffer
- do K Newton steps (K small, e.g., 1..3) starting from previous f estimate
- return f estimate

Notes
-----
- This is a "windowed NR on variable-projection LS" baseline.
- It can be strong vs. ZC on noisy signals, but step response depends on window and NR iters.

PFEBench integration
--------------------
- Inherits BaseEstimator
- Implements tuning_ranges(), reset(), latency_samples, _step()

No external deps beyond numpy.
"""

from __future__ import annotations

from typing import List
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class NewtonRaphsonFrequencyEstimator(BaseEstimator):
    NAME: str = "NewtonRaphsonFrequencyEstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        return [
            TuningParam(
                name="win",
                default=201,
                type="int",
                values=[101, 151, 201, 301],
                description="Window length (samples) for LS fit.",
            ),
            TuningParam(
                name="nr_steps",
                default=2,
                type="int",
                values=[1, 2, 3],
                description="Newton steps per sample (small integer).",
            ),
            TuningParam(
                name="fd_hz",
                default=0.05,
                type="float",
                values=[0.02, 0.05, 0.10],
                description="Finite-difference step (Hz) for g/H approximation.",
            ),
            TuningParam(
                name="ridge",
                default=1e-6,
                type="float",
                values=[0.0, 1e-8, 1e-6],
                description="Ridge regularization for LS (stability).",
            ),
            TuningParam(
                name="ema_alpha",
                default=0.6,
                type="float",
                values=[1.0, 0.6, 0.35],
                description="EMA on output frequency. 1.0 disables smoothing.",
            ),
            TuningParam(
                name="min_freq_hz",
                default=30.0,
                type="float",
                values=[20.0, 30.0, 40.0],
                description="Minimum frequency clamp.",
            ),
            TuningParam(
                name="max_freq_hz",
                default=70.0,
                type="float",
                values=[70.0, 80.0, 90.0],
                description="Maximum frequency clamp.",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._win = int(self._params.get("win", 201))
        self._nr_steps = int(self._params.get("nr_steps", 2))
        self._fd_hz = float(self._params.get("fd_hz", 0.05))
        self._ridge = float(self._params.get("ridge", 1e-6))
        self._ema = float(self._params.get("ema_alpha", 0.6))
        self._min_f = float(self._params.get("min_freq_hz", 30.0))
        self._max_f = float(self._params.get("max_freq_hz", 70.0))

        if self._win < 11:
            self._win = 11
        if self._nr_steps < 1:
            self._nr_steps = 1
        if self._fd_hz <= 0:
            self._fd_hz = 0.05
        if self._ridge < 0:
            self._ridge = 0.0
        if not (0.0 <= self._ema <= 1.0):
            self._ema = 0.6
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0

        self._buf = np.zeros(self._win, dtype=float)
        self._idx = 0
        self._n = 0  # how many samples seen

        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

        # Precompute centered time vector for window (reduces conditioning issues)
        t = np.arange(self._win, dtype=float) / self._fs
        self._t = t - float(np.mean(t))

    @property
    def latency_samples(self) -> int:
        # Windowed estimator: effective delay roughly half window
        return int((self._win - 1) // 2)

    # -----------------------
    # Window handling
    # -----------------------
    def _push(self, x: float) -> None:
        self._buf[self._idx] = float(x)
        self._idx = (self._idx + 1) % self._win
        self._n += 1

    def _window(self) -> np.ndarray:
        # Return oldest..newest
        if self._n < self._win:
            return self._buf[: self._n].copy()
        return np.concatenate((self._buf[self._idx :], self._buf[: self._idx])).copy()

    # -----------------------
    # LS cost for given f
    # -----------------------
    def _sse(self, f_hz: float, xw: np.ndarray) -> float:
        f = float(_clamp(f_hz, self._min_f, self._max_f))
        w = 2.0 * np.pi * f

        # match window length
        t = (
            self._t
            if xw.size == self._win
            else (
                np.arange(xw.size) / self._fs - np.mean(np.arange(xw.size) / self._fs)
            )
        )

        c = np.cos(w * t)
        s = np.sin(w * t)

        # Phi = [cos, sin, 1]
        Phi = np.column_stack([c, s, np.ones_like(c)])

        # ridge LS: theta = (Phi^T Phi + ridge I)^-1 Phi^T x
        G = Phi.T @ Phi
        if self._ridge > 0:
            G = G + self._ridge * np.eye(G.shape[0], dtype=float)
        b = Phi.T @ xw
        try:
            theta = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse
            theta = np.linalg.pinv(G) @ b

        r = xw - Phi @ theta
        return float(r.T @ r)

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        xk = float(v_sample)
        if not np.isfinite(xk):
            return float(self._f_est)

        self._push(xk)

        # Not enough data yet
        if self._n < max(11, self._win // 4):
            return float(self._f_est)

        xw = self._window()

        # Newton iterations on f (Hz) using finite differences of SSE(f)
        f = float(self._f_est)
        h = float(self._fd_hz)

        for _ in range(self._nr_steps):
            f_m = float(_clamp(f - h, self._min_f, self._max_f))
            f_0 = float(_clamp(f, self._min_f, self._max_f))
            f_p = float(_clamp(f + h, self._min_f, self._max_f))

            Jm = self._sse(f_m, xw)
            J0 = self._sse(f_0, xw)
            Jp = self._sse(f_p, xw)

            # g ≈ dJ/df, H ≈ d2J/df2
            g = (Jp - Jm) / (2.0 * h)
            H = (Jp - 2.0 * J0 + Jm) / (h * h)

            if not np.isfinite(g) or not np.isfinite(H) or abs(H) < 1e-12:
                break

            f_new = f - (g / H)

            # Clamp & small trust region (avoid wild jumps)
            f_new = float(_clamp(f_new, self._min_f, self._max_f))
            # trust region: max 2 Hz per sample
            df = float(f_new - f)
            df = float(_clamp(df, -2.0, 2.0))
            f = float(_clamp(f + df, self._min_f, self._max_f))

        # EMA smoothing
        a = float(self._ema)
        if a >= 1.0:
            self._f_est = float(f)
        elif a <= 0.0:
            # freeze
            self._f_est = float(self._f_est)
        else:
            self._f_est = float((1.0 - a) * self._f_est + a * float(f))

        self._f_est = float(_clamp(self._f_est, self._min_f, self._max_f))
        return float(self._f_est)
