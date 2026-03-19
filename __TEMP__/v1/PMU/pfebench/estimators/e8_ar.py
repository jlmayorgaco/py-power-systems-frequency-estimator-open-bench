#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e8_ar.py

E8 — Autoregressive (AR) Frequency Estimator (windowed Burg), strictly online.

What it does
------------
- Maintains a causal buffer of last W samples.
- Each step (once buffer is full):
    1) Fits AR(p) coefficients using Burg's method on the buffer.
    2) Computes roots of AR polynomial A(z) = 1 + a1 z^-1 + ... + ap z^-p.
    3) Picks the dominant oscillatory root (closest to unit circle) within [min_f, max_f].
    4) Converts angle to frequency: f = (fs/(2π)) * angle(root).

Why Burg?
---------
- Stable AR fit (minimizes forward/backward prediction errors).
- Works well for narrowband sinusoids and is robust-ish vs noise.

PFEBench integration
--------------------
- Inherits BaseEstimator
- Implements tuning_ranges(), reset(), latency_samples, _step()
"""

from __future__ import annotations

from collections import deque
from typing import List
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AutoregressiveFrequencyEstimator(BaseEstimator):
    NAME: str = "AutoregressiveFrequencyEstimator"
    FAMILY: str = "Regression-Spectral"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Keep it compact. This method is window-based, so W and p matter most.
        """
        return [
            TuningParam(
                name="ar_order",
                default=6,
                type="int",
                values=[4, 6, 8, 10],
                description="AR model order p. Higher fits more structure, can overfit noise.",
            ),
            TuningParam(
                name="win",
                default=201,
                type="int",
                values=[101, 151, 201, 301],
                description="Window length W (samples) for AR fitting.",
            ),
            TuningParam(
                name="ema_alpha",
                default=0.60,
                type="float",
                values=[1.0, 0.6, 0.35],
                description="EMA smoothing on output frequency. 1.0 = no smoothing.",
            ),
            TuningParam(
                name="min_freq_hz",
                default=40.0,
                type="float",
                values=[30.0, 40.0],
                description="Min plausible frequency bound (Hz).",
            ),
            TuningParam(
                name="max_freq_hz",
                default=80.0,
                type="float",
                values=[70.0, 80.0, 90.0],
                description="Max plausible frequency bound (Hz).",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._p = int(self._params.get("ar_order", 6))
        self._W = int(self._params.get("win", 201))

        self._min_f = float(self._params.get("min_freq_hz", 40.0))
        self._max_f = float(self._params.get("max_freq_hz", 80.0))
        self._ema_alpha = float(self._params.get("ema_alpha", 0.60))

        # sanitize
        if self._p < 2:
            self._p = 2
        if self._W < (2 * self._p + 5):
            self._W = 2 * self._p + 5

        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 0.60
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0

        self._buf = deque(maxlen=self._W)
        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

    @property
    def latency_samples(self) -> int:
        # needs window to fill
        return int(self._W - 1)

    # -----------------------
    # Burg AR(p)
    # -----------------------
    @staticmethod
    def _burg_ar(x: np.ndarray, p: int, eps: float = 1e-12) -> np.ndarray:
        """
        Burg's method for AR coefficients.

        Returns a (length p) array a where:
            x[n] + sum_{k=1..p} a[k-1] x[n-k] = e[n]
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        N = x.size
        if N <= p + 1:
            return np.zeros(p, dtype=float)

        # forward/backward errors
        ef = x[1:].copy()
        eb = x[:-1].copy()

        a = np.zeros(p, dtype=float)

        # total prediction error
        E = float(np.dot(x, x) / N)
        if E < eps:
            return a

        for m in range(1, p + 1):
            # reflection coefficient
            num = -2.0 * float(np.dot(eb, ef))
            den = float(np.dot(ef, ef) + np.dot(eb, eb))
            if den <= eps:
                break
            k = num / den

            # update AR coeffs
            a_prev = a.copy()
            a[m - 1] = k
            if m > 1:
                a[: m - 1] = a_prev[: m - 1] + k * a_prev[: m - 1][::-1]

            # update errors
            ef_new = ef + k * eb
            eb_new = eb + k * ef
            ef = ef_new[1:]
            eb = eb_new[:-1]

            # update prediction error energy
            E *= 1.0 - k * k
            if E <= eps:
                break

        return a

    # -----------------------
    # Root -> frequency
    # -----------------------
    def _ar_to_freq(self, a: np.ndarray) -> float:
        """
        Given AR coeffs a (len p) for:
            x[n] + Σ a[k-1] x[n-k] = e[n]
        Build polynomial in z:
            A(z) = z^p + a1 z^(p-1) + ... + ap
        Roots inside unit circle correspond to modes.
        """
        p = a.size
        if p < 2:
            return float(self._f_est)

        # Polynomial coefficients for z^p + a1 z^(p-1) + ... + ap
        poly = np.r_[1.0, a]  # length p+1
        roots = np.roots(poly)

        # Candidate complex roots with positive imaginary part
        cand = []
        for r in roots:
            if not np.isfinite(r.real) or not np.isfinite(r.imag):
                continue
            if r.imag <= 0:
                continue
            ang = float(np.arctan2(r.imag, r.real))  # (0, π)
            f = ang * self._fs / (2.0 * np.pi)
            if (f >= self._min_f) and (f <= self._max_f):
                # prefer roots closest to unit circle (dominant oscillation)
                rad = float(np.abs(r))
                cand.append((abs(rad - 1.0), -rad, f))

        if not cand:
            return float(self._f_est)

        # sort: closest-to-unit-circle, then higher radius
        cand.sort()
        f_best = float(cand[0][2])
        return float(_clamp(f_best, self._min_f, self._max_f))

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        xk = float(v_sample)
        if not np.isfinite(xk):
            return float(self._f_est)

        self._buf.append(xk)
        if len(self._buf) < self._W:
            return float(self._f_est)

        x = np.asarray(self._buf, dtype=float)

        # optional de-mean (helps DC offset)
        x = x - float(np.mean(x))

        a = self._burg_ar(x, p=self._p, eps=self.EPS)
        f_new = self._ar_to_freq(a)

        # EMA smoothing
        a_ema = float(self._ema_alpha)
        if a_ema >= 1.0:
            self._f_est = float(f_new)
        elif a_ema <= 0.0:
            # keep previous
            pass
        else:
            self._f_est = float((1.0 - a_ema) * self._f_est + a_ema * float(f_new))

        return float(self._f_est)
