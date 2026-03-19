#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e5_zc_ma.py

E5 — Zero-Crossing with Moving Average (MA) Pre-filter (Q1-grade baseline).

What this estimator is
----------------------
A classic zero-crossing (ZC) frequency estimator, but operating on a *smoothed*
signal:

    x_f[n] = MA_w( v[n] )
    detect rising/falling zero crossings on x_f[n]
    estimate half-cycle period from consecutive crossings
    f[n] = fs / (2 * Δn_cross)

Key design goals (benchmark-friendly)
-------------------------------------
- Simple, transparent, and robust baseline (classic DSP).
- Causal streaming API (BaseEstimator): reset(), _step(), latency_samples, tuning_ranges().
- Optional:
  - Linear interpolation for sub-sample crossing time (improves precision).
  - Magnitude gate around crossing to reduce noise-triggered crossings.
  - EMA smoothing on f.
  - Sanity bounds (min/max).

Latency modeling
----------------
Latency is dominated by the MA group delay:
    delay_MA ≈ (w-1)/2 samples

Plus, for half-cycle estimation you only get a new update on each crossing,
but the estimator outputs ZOH/EMA every sample.

Status: Early development – API not stable.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import math
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class ZeroCrossingMAEstimator(BaseEstimator):
    """
    Zero-Crossing estimator with Moving Average (MA) pre-filter.
    """

    NAME: str = "ZeroCrossingMAEstimator"
    FAMILY: str = "TimeDomain-Baseline"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Compact tuning grid (avoid explosion).
        NOTE: Do NOT tune fs; it comes from scenario/runner.
        """
        return [
            TuningParam(
                name="filter_win",
                default=5,
                type="int",
                values=[1, 3, 5, 9, 15],
                description="Causal moving-average window applied to voltage. 1 disables.",
            ),
            TuningParam(
                name="use_interp",
                default=1,
                type="int",
                values=[0, 1],
                description="Use linear interpolation for sub-sample zero-crossing timing.",
            ),
            TuningParam(
                name="min_abs",
                default=0.0,
                type="float",
                values=[0.0, 1e-6, 1e-4],
                description="Magnitude gate: reject crossings when avg(|a|,|b|) < min_abs.",
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
                values=[20.0, 30.0, 40.0],
                description="Minimum plausible frequency for outlier rejection (Hz).",
            ),
            TuningParam(
                name="max_freq_hz",
                default=80.0,
                type="float",
                values=[65.0, 80.0, 120.0],
                description="Maximum plausible frequency for outlier rejection (Hz).",
            ),
            TuningParam(
                name="f_nom_hz",
                default=cls.NOMINAL_FREQ_HZ,
                type="float",
                values=[60.0],
                description="Nominal frequency (initialization only; keep fixed).",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._filter_win = max(1, int(self._params.get("filter_win", 1)))
        self._use_interp = bool(int(self._params.get("use_interp", 1)))
        self._min_abs = float(self._params.get("min_abs", 0.0))

        self._ema_alpha = float(self._params.get("ema_alpha", 0.65))
        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 0.65

        self._min_f = float(self._params.get("min_freq_hz", 40.0))
        self._max_f = float(self._params.get("max_freq_hz", 80.0))
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0

        # MA buffer (oldest->newest)
        self._xbuf: deque = deque(maxlen=self._filter_win)

        # Crossing tracking
        self._prev_val: float = 0.0  # filtered prev sample
        self._total_samples: int = 0

        # store last crossing time (fractional absolute sample index)
        self._last_cross_abs: Optional[float] = None
        self._have_first_cross: bool = False

        # Output
        self._f_est: float = float(self._f_nom)

    @property
    def latency_samples(self) -> int:
        # group delay of MA (causal)
        return int(round((self._filter_win - 1) / 2.0))

    # -----------------------
    # Helpers
    # -----------------------
    def _ma(self, x: float) -> float:
        self._xbuf.append(float(x))
        if self._filter_win == 1:
            return float(x)
        return float(sum(self._xbuf) / len(self._xbuf))

    def _valid_crossing_pair(self, a: float, b: float) -> bool:
        return (0.5 * (abs(a) + abs(b))) >= self._min_abs

    def _interp_zero_crossing(self, a: float, b: float) -> float:
        """
        Linear interpolation fraction for zero-crossing between samples:
            a at n-1, b at n, find frac in [0,1] such that a + frac*(b-a) = 0
        """
        denom = a - b
        if abs(denom) < self.EPS:
            return 0.0
        frac = a / denom  # a/(a-b)
        if frac < 0.0:
            return 0.0
        if frac > 1.0:
            return 1.0
        return float(frac)

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        self._total_samples += 1

        # 1) Pre-filter with MA
        x = self._ma(v_sample)

        # 2) Detect any zero-crossing (sign change)
        prev = self._prev_val
        crossed = (prev <= 0.0 < x) or (prev >= 0.0 > x)

        if crossed and self._valid_crossing_pair(prev, x):
            # absolute sample index of current sample is (_total_samples-1)
            # previous is (_total_samples-2)
            frac = self._interp_zero_crossing(prev, x) if self._use_interp else 1.0
            cross_abs = (self._total_samples - 2) + frac

            # 3) If we have a previous crossing, estimate half-cycle
            if self._have_first_cross and (self._last_cross_abs is not None):
                dt_samples = cross_abs - self._last_cross_abs

                # half-cycle period (samples) -> full period = 2*dt
                if dt_samples > self.EPS:
                    f_new = self._fs / (2.0 * dt_samples)

                    # sanity bounds
                    if math.isfinite(f_new) and (self._min_f < f_new < self._max_f):
                        a = self._ema_alpha
                        if a <= 0.0:
                            pass  # hold
                        elif a >= 1.0:
                            self._f_est = float(f_new)
                        else:
                            self._f_est = float(
                                (1.0 - a) * self._f_est + a * float(f_new)
                            )

            # update last crossing
            self._last_cross_abs = cross_abs
            self._have_first_cross = True

        # 4) update state
        self._prev_val = x

        # 5) output
        if not math.isfinite(self._f_est):
            return float("nan")
        return float(self._f_est)
