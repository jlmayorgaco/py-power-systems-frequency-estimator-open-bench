"""
openfreqbench/estimators/zc.py

Zero-Crossing Frequency Estimator with sub-sample linear interpolation.

Algorithm:
  1. Moving-average pre-filter (window = filter_win).
  2. Detect positive-to-negative or negative-to-positive zero crossings.
  3. Sub-sample interpolation: delta = |v_prev| / (|v_prev| + |v_curr|).
  4. Half-cycle period → f = 1 / (2 * T_half).
  5. Outlier rejection: discard estimates outside [40, 80] Hz.

Ported from pfebench/estimators/e1_zc.py (pfebench research system).
Timing removed — measured externally by TimingHarness.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from openfreqbench.estimators._base import BaseEstimator, TuningParam


class ZeroCrossingEstimator(BaseEstimator):
    """
    Time-domain baseline: zero-crossing with moving-average pre-filter.

    Tunable parameter
    -----------------
    filter_win : int
        Moving-average window length. Default 5.
        Larger → more noise rejection, more latency.
    """

    NAME = "ZeroCrossing"
    FAMILY = "TimeDomain"

    MIN_VALID_FREQ_HZ: float = 40.0
    MAX_VALID_FREQ_HZ: float = 80.0
    NOMINAL_FREQ_HZ: float = 60.0
    _EPSILON: float = 1e-9

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        return [
            TuningParam(
                name="filter_win",
                default=5,
                type="int",
                values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 35, 50, 100, 120, 250],
                description="Moving-average pre-filter window size.",
            )
        ]

    def reset(self) -> None:
        self._fs: float = float(self._params.get("fs", 10_000.0))
        self._win_len: int = int(self._params.get("filter_win", 5))
        self._buffer: deque = deque(maxlen=self._win_len)
        self._prev_val: float = 0.0
        self._total_samples: int = 0
        self._last_cross_absolute: float = 0.0
        self._first_cross_detected: bool = False
        self._f_est: float = self.NOMINAL_FREQ_HZ

    @property
    def latency_samples(self) -> int:
        """Filter group delay + half half-cycle measurement delay."""
        filter_delay = (self._win_len - 1) / 2.0
        zc_delay = self._fs / (4.0 * self.NOMINAL_FREQ_HZ)
        return int(filter_delay + zc_delay)

    def _step(self, v_sample: float) -> float:
        # 1. Pre-filter
        self._buffer.append(v_sample)
        if len(self._buffer) < self._win_len:
            return self.NOMINAL_FREQ_HZ

        v_filt = sum(self._buffer) / len(self._buffer)

        # 2. Zero-crossing detection
        curr_sign = np.sign(v_filt)
        prev_sign = np.sign(self._prev_val)

        if (curr_sign != prev_sign) and (self._total_samples > 0):
            self._handle_crossing(v_filt)

        self._prev_val = v_filt
        self._total_samples += 1
        return self._f_est

    def _handle_crossing(self, v_curr: float) -> None:
        # 3. Sub-sample interpolation
        abs_prev = abs(self._prev_val)
        abs_curr = abs(v_curr)
        denom = abs_prev + abs_curr
        delta = abs_prev / denom if denom > self._EPSILON else 0.5

        current_cross_absolute = (self._total_samples - 1) + delta

        if self._first_cross_detected:
            samples_diff = current_cross_absolute - self._last_cross_absolute
            if samples_diff > self._EPSILON:
                t_half = samples_diff / self._fs
                new_f = 1.0 / (2.0 * t_half)
                # 4. Outlier rejection
                if self.MIN_VALID_FREQ_HZ < new_f < self.MAX_VALID_FREQ_HZ:
                    self._f_est = new_f

        self._last_cross_absolute = current_cross_absolute
        self._first_cross_detected = True
