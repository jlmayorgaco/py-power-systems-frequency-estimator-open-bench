"""
pfebench/estimators/e1_zc.py

Zero-Crossing Frequency Estimator with Linear Interpolation.
A time-domain baseline algorithm characterized by low computational cost.

Key Features:
  - Sub-sample Interpolation: Achieves precision exceeding the sampling rate.
  - Tunable Pre-filter: Moving average window to mitigate Gaussian noise.
  - Dynamic Latency Reporting: Reports delay based on filter window size.
"""

from __future__ import annotations
from collections import deque
from typing import List, Any, Dict, Optional

import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class ZeroCrossingEstimator(BaseEstimator):
    """
    Implements a zero-crossing frequency estimator augmented with linear interpolation
    and a moving average pre-filter.
    """

    NAME = "ZeroCrossing"
    FAMILY = "TimeDomain"

    # Domain Constants
    MIN_VALID_FREQ_HZ = 40.0
    MAX_VALID_FREQ_HZ = 80.0
    NOMINAL_FREQ_HZ = 60.0
    EPSILON = 1e-9

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Defines the hyperparameter search space for Grid Search Optimization (GSO).
        """
        return [
            TuningParam(
                name="filter_win",
                default=5,
                type="int",
                values=[
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    15,
                    20,
                    25,
                    35,
                    50,
                    100,
                    120,
                    250,
                ],
                description="Window size for the Moving Average pre-filter.",
            )
        ]

    def reset(self) -> None:
        """Resets the internal state of the estimator."""
        # Configuration
        self._fs = float(self._params.get("fs", 1000.0))
        self._win_len = int(self._params.get("filter_win", 5))

        # Signal Buffer
        self._buffer: deque = deque(maxlen=self._win_len)

        # Zero-Crossing State
        self._prev_val = 0.0
        self._total_samples = 0
        self._last_cross_absolute = 0.0
        self._first_cross_detected = False

        # Output State (Zero-Order Hold)
        self._f_est = self.NOMINAL_FREQ_HZ

    @property
    def latency_samples(self) -> int:
        """
        Estimates total latency: Group delay of the filter + Intrinsic ZC delay.

        Calculation:
        - MA Filter: (N-1)/2 samples.
        - ZC Method: Averages the last half-cycle (~8.33ms at 60Hz).
          The measurement center of mass is effectively half that duration (~4.16ms).
          4.16ms corresponds to approx 1/4 of a 60Hz cycle (fs / 240.0).
        """
        filter_delay = (self._win_len - 1) / 2.0
        zc_delay = self._fs / (4.0 * self.NOMINAL_FREQ_HZ)
        return int(filter_delay + zc_delay)

    def _step(self, v_sample: float) -> float:
        """
        Processes a single voltage sample.
        Returns the latest frequency estimate (Zero-Order Hold).
        """
        # 1. Pre-filtering (Moving Average)
        self._buffer.append(v_sample)

        # Warm-up phase
        if len(self._buffer) < self._win_len:
            return self.NOMINAL_FREQ_HZ

        # Compute moving average
        v_filt = sum(self._buffer) / len(self._buffer)

        # 2. Zero-Crossing Detection
        curr_sign = np.sign(v_filt)
        prev_sign = np.sign(self._prev_val)

        # Check for sign change (crossing)
        # Note: We check total_samples > 0 to ensure we have a valid previous value
        if (curr_sign != prev_sign) and (self._total_samples > 0):
            self._handle_crossing(v_filt)

        # Update state
        self._prev_val = v_filt
        self._total_samples += 1

        return self._f_est

    def _handle_crossing(self, v_curr: float) -> None:
        """
        Handles logic upon detecting a zero-crossing.
        Computes exact crossing time via linear interpolation and updates frequency estimate.
        """
        # 3. Linear Interpolation (Sub-sample accuracy)
        # We determine the fraction 'delta' of the sampling interval where the crossing occurred.
        # Formula: delta = |v_prev| / (|v_prev| + |v_curr|)
        try:
            abs_prev = abs(self._prev_val)
            abs_curr = abs(v_curr)
            denom = abs_prev + abs_curr

            if denom > self.EPSILON:
                delta = abs_prev / denom
            else:
                delta = 0.5  # Boundary case: adjacent zero values
        except ZeroDivisionError:
            delta = 0.5

        current_cross_absolute = (self._total_samples - 1) + delta

        if self._first_cross_detected:
            # Calculate duration since last crossing (Half-Cycle duration)
            samples_diff = current_cross_absolute - self._last_cross_absolute

            if samples_diff > self.EPSILON:
                t_half_cycle = samples_diff / self._fs
                # Frequency = 1 / Period = 1 / (2 * t_half_cycle)
                new_f = 1.0 / (2.0 * t_half_cycle)

                # 4. Outlier Rejection / Sanity Check
                if self.MIN_VALID_FREQ_HZ < new_f < self.MAX_VALID_FREQ_HZ:
                    self._f_est = new_f

        # Update crossing state
        self._last_cross_absolute = current_cross_absolute
        self._first_cross_detected = True
