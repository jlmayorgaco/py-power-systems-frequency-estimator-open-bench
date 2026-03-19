#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e9_prony.py

E9 — Prony Method (Time-Domain) Frequency Estimator (strictly causal, moving-window).

What it does
------------
Uses a classic Prony / linear-prediction (AR) fit on a sliding window of raw samples,
then converts the dominant complex root into frequency.

For a (near) single-tone sinusoid, an AR(2) model is enough:

    x[n] + a1 x[n-1] + a2 x[n-2] ≈ 0

Estimate [a1,a2] by least squares on a window, then solve the characteristic polynomial:
    z^2 + a1 z + a2 = 0

If roots are z = r e^{±jω}, then ω = angle(z) and:
    f = (fs / (2π)) * ω

This is "Prony-like" in the time domain (linear prediction + root finding).
It is causal: at time n it uses only the last W samples.

PFEBench integration
--------------------
- Inherits BaseEstimator
- Implements tuning_ranges(), reset(), latency_samples, _step()
- Params are set via BaseEstimator.set_params(**kwargs)

Notes / expectations
--------------------
- Very good on clean single-tone; degrades with strong harmonics/interharmonics.
- Unlike E7_RLS_BASIC, this re-fits from scratch on each window (LS), so it can be
  more stable on steps but more expensive per-sample.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class PronyMethodEstimator(BaseEstimator):
    NAME: str = "PronyMethodEstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0

    EPS: float = 1e-12
    DEN_EPS: float = 1e-18

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Compact grid; fs comes from scenario/runner, do NOT tune it.
        ar_order=2 is the "classic" single-tone Prony (dominant complex pair).
        """
        return [
            TuningParam(
                name="win",
                default=201,
                type="int",
                values=[101, 151, 201, 301],
                description="Window length W (samples) for Prony/LP fit. Must be >= ar_order+1.",
            ),
            TuningParam(
                name="ar_order",
                default=2,
                type="int",
                values=[2, 4],
                description="Linear prediction order. 2 for single tone; 4 can help mild distortion.",
            ),
            TuningParam(
                name="ridge",
                default=1e-6,
                type="float",
                values=[0.0, 1e-8, 1e-6],
                description="Small ridge regularization for the normal equations (stability).",
            ),
            TuningParam(
                name="ema_alpha",
                default=0.60,
                type="float",
                values=[1.0, 0.60, 0.35],
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
            TuningParam(
                name="min_r",
                default=0.80,
                type="float",
                values=[0.50, 0.70, 0.80, 0.90],
                description="Minimum root magnitude r to accept (reject overly damped/garbage roots).",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._win = int(self._params.get("win", 201))
        self._p = int(self._params.get("ar_order", 2))
        self._ridge = float(self._params.get("ridge", 1e-6))
        self._ema_alpha = float(self._params.get("ema_alpha", 0.60))

        self._min_f = float(self._params.get("min_freq_hz", 40.0))
        self._max_f = float(self._params.get("max_freq_hz", 80.0))
        self._min_r = float(self._params.get("min_r", 0.80))

        # sanitize
        if self._p < 1:
            self._p = 2
        if self._win < self._p + 2:
            self._win = self._p + 2
        if self._ridge < 0.0:
            self._ridge = 0.0
        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 0.60
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0
        if self._min_r < 0.0:
            self._min_r = 0.0
        if self._min_r > 1.5:
            self._min_r = 1.5

        self._buf = deque(maxlen=self._win)
        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

    @property
    def latency_samples(self) -> int:
        # Causal window (estimate timestamped at "now"), so true latency is 0.
        # (Warmup is handled by returning previous until buffer is full.)
        return 0

    # -----------------------
    # Core math
    # -----------------------
    def _fit_lp_coeffs(self, x: np.ndarray) -> np.ndarray:
        """
        Fit linear prediction coefficients a (size p):
          x[n] + Σ_{i=1..p} a[i-1] x[n-i] ≈ 0
        Over n = p..N-1 (LS).

        Build:
          y = -x[p:]
          Phi rows = [x[n-1], x[n-2], ..., x[n-p]]
        Solve (Phi^T Phi + ridge I) a = Phi^T y
        """
        N = x.size
        p = self._p
        y = -x[p:]  # shape (N-p,)

        # Phi: (N-p) x p
        Phi = np.empty((N - p, p), dtype=float)
        for i in range(p):
            Phi[:, i] = x[p - 1 - i : N - 1 - i]

        # Normal equations with ridge
        G = Phi.T @ Phi
        if self._ridge > 0.0:
            G = G + float(self._ridge) * np.eye(p, dtype=float)
        b = Phi.T @ y

        # Solve robustly
        try:
            a = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(G, b, rcond=None)[0]
        return np.asarray(a, dtype=float)

    def _roots_to_freq(self, roots: np.ndarray) -> float:
        """
        Pick a plausible oscillatory root (complex) and convert to frequency.
        Strategy:
          - take roots with positive imaginary part (angle in (0, pi))
          - prefer magnitude closest to 1 (sinusoid)
          - reject if magnitude < min_r
        """
        if roots.size == 0:
            return float(self._f_est)

        best = None
        best_score = float("inf")

        for z in roots:
            if not np.isfinite(z.real) or not np.isfinite(z.imag):
                continue
            if z.imag <= 0.0:
                continue
            r = float(np.abs(z))
            if r < self._min_r:
                continue
            # score: closeness to unit circle
            score = abs(r - 1.0)
            if score < best_score:
                best_score = score
                best = z

        if best is None:
            return float(self._f_est)

        omega = float(np.angle(best))  # (0, pi)
        f = omega * self._fs / (2.0 * np.pi)

        if not np.isfinite(f):
            return float(self._f_est)

        # Bounds/outlier reject
        if (f < self._min_f) or (f > self._max_f):
            return float(self._f_est)

        return float(f)

    # -----------------------
    # Online step
    # -----------------------
    def _step(self, v_sample: float) -> float:
        xk = float(v_sample)
        if not np.isfinite(xk):
            return float(self._f_est)

        self._buf.append(xk)

        # Warmup
        if len(self._buf) < self._win:
            return float(self._f_est)

        x = np.asarray(self._buf, dtype=float)

        # Remove mean (helps with DC bias)
        x = x - float(np.mean(x))

        # Fit LP/Prony coefficients
        a = self._fit_lp_coeffs(x)

        # Polynomial: z^p + a1 z^(p-1) + ... + ap = 0
        poly = np.concatenate(([1.0], a.astype(float)))
        roots = np.roots(poly)

        f_new = self._roots_to_freq(roots)

        # EMA smoothing
        a_ema = float(self._ema_alpha)
        if a_ema <= 0.0:
            return float(self._f_est)
        if a_ema >= 1.0:
            self._f_est = float(f_new)
        else:
            self._f_est = float((1.0 - a_ema) * self._f_est + a_ema * float(f_new))

        self._f_est = float(_clamp(self._f_est, self._min_f, self._max_f))
        return float(self._f_est)
