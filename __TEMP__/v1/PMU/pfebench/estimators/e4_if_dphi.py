#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e4_if_dphi.py

E4 — Instantaneous Frequency via Phase Increment (IF-Δφ) (Q1-grade baseline).

Core idea
---------
Estimate instantaneous frequency from the phase increment of the analytic signal:

    z[n] = x_I[n] + j x_Q[n]         (analytic signal)
    φ[n] = atan2(x_Q[n], x_I[n])
    Δφ[n] = wrap( φ[n] - φ[n-1] )
    f[n]  = (fs / (2π)) * Δφ[n]

Implementation choices (benchmark-friendly)
-------------------------------------------
- Optional causal moving-average on input x[n]
- FIR Hilbert transformer (odd length, windowed) to approximate quadrature x_Q[n]
- Align in-phase x_I with FIR group delay M=(L-1)/2
- Optional moving-average on Δφ
- Optional EMA smoothing on f
- Magnitude gate (min_abs)
- Sanity bounds (min/max frequency) with outlier rejection

Why you saw ~55 Hz with fs=10k, f≈60
------------------------------------
A short windowed Hilbert (31/51/81) is a poor approximation very near DC.
At fs=10k, 60 Hz is normalized frequency 0.006 → needs a much longer FIR
to behave like a proper 90° phase shifter. Otherwise φ and Δφ bias badly.

Fixes included here
-------------------
1) Keep the DSP correctness fixes you already have:
   - causal FIR: dot(h, xb[::-1])
   - I/Q alignment: I uses buffer index M

2) Make tuning grid fs-aware and MUCH smaller:
   - Provide only sensible hilbert_len options for common fs regimes
   - Reduce combinations drastically
   - Avoid tuning min_abs in clean synthetic cases (set to 0 by default)

This keeps runtime sane and avoids searching in "dead" parameter regions.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import math
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class InstantaneousFrequencyPhaseIncrementEstimator(BaseEstimator):
    """
    IF-Δφ estimator: instantaneous frequency via phase increment of analytic signal.
    """

    NAME: str = "InstantaneousFrequencyPhaseIncrementEstimator"
    FAMILY: str = "TimeDomain-Analytic"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    # -----------------------
    # Tuning grid (compact + fs-aware)
    # -----------------------
    @staticmethod
    def _hilbert_len_grid(fs_hz: float) -> List[int]:
        """
        Practical, compact candidate set.

        - For fs ~ 1k: 81 or 121 works well for 50/60 Hz.
        - For fs ~ 10k: need longer filters to behave near DC -> 401/801.
        """
        fs = float(fs_hz)

        if fs <= 1200.0:
            return [81, 121]  # compact, works well
        if fs <= 3000.0:
            return [161, 241]  # intermediate
        if fs <= 12000.0:
            return [401, 801]  # typical for your 10k scenarios
        return [801, 1201]  # very high fs

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        IMPORTANT:
        - Do NOT tune `fs` (comes from scenario / runner config).
        - Do NOT tune `f_nom_hz` (only initialization).

        Keep grid compact.
        We set hilbert_len values based on a *typical* fs.
        If you use multiple fs across scenarios, that's fine: you can also override
        hilbert_len directly per method config; or (recommended) rely on the adaptive
        reset() below which clamps hilbert_len to a good choice if user passes an odd bad value.

        Grid size (default):
          filter_win(3) * hilbert_len(2) * dphi_win(3) * ema_alpha(2) * min/max (1*1) = 36 combos
        """
        # Default assumption: many of your scenarios are fs=10k
        hilb_vals = cls._hilbert_len_grid(10_000.0)

        return [
            TuningParam(
                name="filter_win",
                default=1,
                type="int",
                values=[1, 3, 5],
                description="Input moving-average pre-filter window. 1 disables.",
            ),
            TuningParam(
                name="hilbert_len",
                default=hilb_vals[0],
                type="int",
                values=hilb_vals,
                description="Odd length of FIR Hilbert transformer (fs-aware defaults).",
            ),
            TuningParam(
                name="dphi_win",
                default=1,
                type="int",
                values=[1, 3, 5],
                description="Moving-average window on phase increment Δφ. 1 disables.",
            ),
            TuningParam(
                name="min_abs",
                default=0.0,
                type="float",
                values=[0.0],  # keep fixed; tune only if you truly need gating
                description="Minimum analytic magnitude |z| gate to accept updates.",
            ),
            TuningParam(
                name="ema_alpha",
                default=1.0,
                type="float",
                values=[1.0, 0.35],
                description="EMA smoothing on frequency. 1.0 = no smoothing.",
            ),
            # For PFEBench, bounds are usually scenario-driven; keep fixed here.
            TuningParam(
                name="min_freq_hz",
                default=30.0,
                type="float",
                values=[30.0],
                description="Minimum plausible frequency (Hz) for outlier rejection.",
            ),
            TuningParam(
                name="max_freq_hz",
                default=90.0,
                type="float",
                values=[90.0],
                description="Maximum plausible frequency (Hz) for outlier rejection.",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        # fs comes from config/runner; keep here but DO NOT tune
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))

        # only for initialization; keep here but DO NOT tune
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._filter_win = max(1, int(self._params.get("filter_win", 1)))
        self._hilbert_len = int(self._params.get("hilbert_len", 81))
        self._dphi_win = max(1, int(self._params.get("dphi_win", 1)))

        self._min_abs = float(self._params.get("min_abs", 0.0))
        self._ema_alpha = float(self._params.get("ema_alpha", 1.0))
        self._min_f = float(self._params.get("min_freq_hz", 30.0))
        self._max_f = float(self._params.get("max_freq_hz", 90.0))

        # enforce odd hilbert_len >= 3
        if self._hilbert_len < 3:
            self._hilbert_len = 3
        if self._hilbert_len % 2 == 0:
            self._hilbert_len += 1

        # fs-aware sanity: if someone gave a tiny hilbert_len at huge fs, clamp it upward
        # (prevents the classic "55 Hz" bias problem at fs=10k)
        good_grid = self._hilbert_len_grid(self._fs)
        if self._fs >= 8000.0 and self._hilbert_len < good_grid[0]:
            self._hilbert_len = int(good_grid[0])

        # clamp EMA alpha
        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 1.0

        # Buffers (oldest -> newest)
        self._xbuf = deque(maxlen=self._filter_win)
        self._hbuf = deque(maxlen=self._hilbert_len)
        self._ibuf = deque(maxlen=self._hilbert_len)
        self._dphibuf = deque(maxlen=self._dphi_win)

        # FIR Hilbert coeffs
        self._h = self._design_hilbert_fir(self._hilbert_len)
        self._M = (self._hilbert_len - 1) // 2

        # State
        self._phi_prev: Optional[float] = None
        self._f_est = float(self._f_nom)

    @property
    def latency_samples(self) -> int:
        d_in = (self._filter_win - 1) / 2.0
        d_h = (self._hilbert_len - 1) / 2.0
        d_dphi = (self._dphi_win - 1) / 2.0
        return int(round(d_in + d_h + d_dphi))

    # -----------------------
    # DSP helpers
    # -----------------------
    def _ma_in(self, x: float) -> float:
        self._xbuf.append(float(x))
        if self._filter_win == 1:
            return float(x)
        return float(sum(self._xbuf) / len(self._xbuf))

    def _ma_dphi(self, dphi: float) -> float:
        self._dphibuf.append(float(dphi))
        if self._dphi_win == 1:
            return float(dphi)
        return float(sum(self._dphibuf) / len(self._dphibuf))

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
            if n == 0:
                h[k] = 0.0
            elif n % 2 == 0:
                h[k] = 0.0
            else:
                h[k] = 2.0 / (np.pi * n)

        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / (L - 1))
        h *= w

        # zero-DC numerical tweak
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
        return float(np.dot(self._h, xb[::-1]))  # causal mapping

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        # NaN/Inf input: hold
        if not math.isfinite(v_sample):
            return float(self._f_est)

        # 1) Optional input MA
        x = self._ma_in(v_sample)

        # 2) Quadrature via Hilbert FIR (delayed by M)
        q = self._hilbert_quadrature(x)

        # 3) Delay line for in-phase alignment
        self._ibuf.append(float(x))
        if len(self._ibuf) < self._hilbert_len:
            return float(self._f_est)

        # q[n] aligns with x[n-M] => index M in oldest..newest buffer
        I = float(self._ibuf[self._M])
        Q = float(q)

        # 4) Magnitude gate
        mag = math.hypot(I, Q)
        if mag < self._min_abs:
            return float(self._f_est)

        # 5) Phase & increment
        phi = float(np.arctan2(Q, I))
        if self._phi_prev is None:
            self._phi_prev = phi
            return float(self._f_est)

        dphi = self._wrap_to_pi(phi - self._phi_prev)
        self._phi_prev = phi

        # 6) Optional Δφ smoothing
        dphi_eff = self._ma_dphi(dphi)

        # 7) Convert to frequency
        f_new = float((self._fs / (2.0 * np.pi)) * dphi_eff)

        # Robust sign convention
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
