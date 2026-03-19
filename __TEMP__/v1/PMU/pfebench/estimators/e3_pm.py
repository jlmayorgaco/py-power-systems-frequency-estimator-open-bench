#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e3_pm.py

E3 — Period Measurement (PM) Frequency Estimator (Q1-grade baseline).

What this estimator is
----------------------
A classic time-domain frequency estimator that measures the full-cycle period
between consecutive rising (negative→positive) zero-crossings:

    f[k] = fs / (n_rise[k] - n_rise[k-1])

Key design choices (benchmark-friendly)
---------------------------------------
1) Sub-sample interpolation:
   - Linear interpolation around the zero-crossing for fractional-sample timing.

2) Robustness knobs (tuning-ready):
   - Optional moving-average pre-filter (MA) to mitigate noise.
   - Minimum amplitude gate to avoid false crossings near 0 under noise.
   - Sanity bounds (min/max frequency).
   - Optional exponential smoothing of the measured frequency.

3) Causal streaming API:
   - Implements BaseEstimator: reset(), _step(), latency_samples, tuning_ranges().

Latency modeling (for alignment)
--------------------------------
A period measurement computed at the second rising crossing corresponds to the
midpoint of the measured interval. Therefore, the estimate is effectively centered
~T/2 in the past. We report:

    latency ≈ group_delay(MA) + fs / (2*f_nom)

Notes
-----
- This PM baseline is intentionally conservative and transparent.
- IMPORTANT (runner efficiency): Do NOT tune fs or f_nom_hz here.
  They come from the scenario/runner configuration and would explode the grid.

Author: pfebench
Status: Early development – API not stable.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import math
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class PeriodMeasurementEstimator(BaseEstimator):
    """
    Period Measurement (PM) estimator using rising zero-crossings.

    Output:
      - Instantaneous frequency estimate in Hz (zero-order-hold between crossings).
    """

    NAME: str = "PeriodMeasurementEstimator"
    FAMILY: str = "TimeDomain-Baseline"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0

    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Compact Q1-grade tuning grid (avoid combinatorial explosion).

        Tuned knobs:
          - filter_win: MA window length (1 disables)
          - min_abs: amplitude gate for crossing validity
          - ema_alpha: smoothing factor
          - min_freq_hz / max_freq_hz: sanity bounds

        NOT tuned:
          - fs: set by scenario/runner (but still read in reset)
          - f_nom_hz: only for latency approximation (but still read in reset)
        """
        return [
            TuningParam(
                name="filter_win",
                default=1,
                type="int",
                values=[1, 3, 5, 9],
                description="Moving-average pre-filter window. 1 = no filter.",
            ),
            TuningParam(
                name="min_abs",
                default=1e-4,
                type="float",
                values=[0.0, 1e-6, 1e-4],
                description="Minimum avg |v| around candidate crossing to accept it (noise gate).",
            ),
            TuningParam(
                name="ema_alpha",
                default=0.35,
                type="float",
                values=[1.0, 0.65, 0.35],
                description="EMA smoothing on f_new. 1.0 = no smoothing.",
            ),
            TuningParam(
                name="min_freq_hz",
                default=45.0,
                type="float",
                values=[20.0, 30.0],
                description="Minimum plausible frequency for outlier rejection (Hz).",
            ),
            TuningParam(
                name="max_freq_hz",
                default=75.0,
                type="float",
                values=[65.0, 80.0],
                description="Maximum plausible frequency for outlier rejection (Hz).",
            ),
        ]

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        # NOTE: fs & f_nom are provided by runner/scenario (do not tune them).
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._win_len = max(1, int(self._params.get("filter_win", 1)))

        self._min_abs = float(self._params.get("min_abs", 1e-4))
        self._ema_alpha = float(self._params.get("ema_alpha", 0.35))
        self._min_f = float(self._params.get("min_freq_hz", 45.0))
        self._max_f = float(self._params.get("max_freq_hz", 75.0))

        # Clamp alpha safely
        if not (0.0 <= self._ema_alpha <= 1.0):
            self._ema_alpha = 0.35

        # Ensure bounds are sane
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0

        # Filter buffer
        self._buffer: deque = deque(maxlen=self._win_len)

        # Crossing state
        self._prev_val: float = 0.0
        self._total_samples: int = 0

        # Rising-crossing timestamps (fractional sample index)
        self._last_rise_abs: Optional[float] = None
        self._have_first_rise: bool = False

        # Output (ZOH between crossings)
        self._f_est: float = float(self._f_nom)

    @property
    def latency_samples(self) -> int:
        """
        Latency model: group delay of MA + half nominal period.
        Returned as an integer number of samples (rounded).
        """
        filter_delay = (self._win_len - 1) / 2.0
        pm_delay = self._fs / (2.0 * max(self._f_nom, self.EPS))
        return int(round(filter_delay + pm_delay))

    # -----------------------
    # Internal helpers
    # -----------------------
    def _ma(self, x: float) -> float:
        """Causal moving-average (streaming)."""
        self._buffer.append(float(x))
        if self._win_len == 1:
            return float(x)
        return float(sum(self._buffer) / len(self._buffer))

    def _valid_crossing_pair(self, a: float, b: float) -> bool:
        """Reject noise-driven crossings when both samples are near 0."""
        return (0.5 * (abs(a) + abs(b))) >= self._min_abs

    def _interp_zero_crossing(self, a: float, b: float) -> float:
        """
        Linear interpolation fraction for zero-crossing between samples (a at n-1, b at n).
        Returns frac in [0, 1] such that: a + frac*(b-a) = 0
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
        """Streaming step: update filter, detect rising crossing, update period/frequency."""
        self._total_samples += 1

        # 1) Pre-filter
        v = self._ma(v_sample)

        # 2) Rising zero-crossing detection (prev <= 0 and v > 0)
        prev = self._prev_val
        rising = (prev <= 0.0) and (v > 0.0)

        if rising and self._valid_crossing_pair(prev, v):
            # Fractional crossing location between (n-1) and n in absolute sample index.
            # _total_samples is 1-based count; current sample abs idx = _total_samples-1
            # previous sample abs idx = _total_samples-2
            frac = self._interp_zero_crossing(prev, v)
            cross_abs = (self._total_samples - 2) + frac

            if self._have_first_rise and (self._last_rise_abs is not None):
                period_samples = cross_abs - self._last_rise_abs

                if period_samples > self.EPS:
                    f_new = self._fs / period_samples

                    # Sanity / outlier rejection
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

            # Update rising-crossing reference
            self._last_rise_abs = cross_abs
            self._have_first_rise = True

        # 3) Update state
        self._prev_val = v

        # 4) Output (zero-order hold)
        if not math.isfinite(self._f_est):
            return float("nan")
        return float(self._f_est)
