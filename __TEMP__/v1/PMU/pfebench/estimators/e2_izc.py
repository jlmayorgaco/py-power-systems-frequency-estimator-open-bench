#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e2_izc.py

Interpolated Zero-Crossing (IZC) Frequency Estimator.
Time-domain baseline focused on sub-sample zero-crossing interpolation.

Difference vs e1_zc (your current ZC):
- IZC here is the "pure" interpolated zero-crossing baseline:
  * No pre-filter (or optional minimal pre-filter via tuning).
  * No extra "intrinsic ZC delay" modeling: latency is mainly from optional filter.
  * Uses BOTH rising + falling zero-crossings (half-cycle intervals).

Rationale for benchmark:
- Keeps the method minimal so its failure modes are attributable to the ZC principle,
  not to additional smoothing/heuristics.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


class InterpolatedZeroCrossingEstimator(BaseEstimator):
    """
    Pure Interpolated Zero-Crossing (IZC) frequency estimator.

    - Detect sign changes on v_filt (optionally filtered).
    - Estimate crossing time with linear interpolation.
    - Compute frequency from consecutive crossings (half-cycle).
    - Output is zero-order held between crossings.

    Notes:
    - This baseline is intentionally fragile under noise/harmonics.
    - It provides a clean "time-domain lower bound" for your benchmark.
    """

    NAME = "InterpolatedZeroCrossing"
    FAMILY = "TimeDomain"

    # Domain constants
    MIN_VALID_FREQ_HZ = 40.0
    MAX_VALID_FREQ_HZ = 80.0
    NOMINAL_FREQ_HZ = 60.0
    EPSILON = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Minimal tuning: optional moving-average window.
        Set filter_win=1 for "pure" IZC (no filtering).
        """
        return [
            TuningParam(
                name="filter_win",
                default=1,
                type="int",
                values=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 35, 50, 100, 120, 250],
                description="Optional MA pre-filter window. 1 = no filter (pure IZC).",
            )
        ]

    def reset(self) -> None:
        # Configuration
        self._fs = float(self._params.get("fs", 1000.0))
        self._win_len = int(self._params.get("filter_win", 1))
        self._win_len = max(1, self._win_len)

        # Optional pre-filter buffer (MA)
        self._buffer: deque = deque(maxlen=self._win_len)

        # ZC state
        self._prev_val: float = 0.0
        self._total_samples: int = 0

        # crossing times (absolute sample index + fractional delta)
        self._last_cross_absolute: float = 0.0
        self._first_cross_detected: bool = False

        # Output (ZOH)
        self._f_est: float = self.NOMINAL_FREQ_HZ

    @property
    def latency_samples(self) -> int:
        """
        For this "pure" IZC baseline:
        - Latency is dominated by the optional MA filter group delay.
        - The estimator updates on each crossing; we do NOT add extra heuristic delay.
        """
        filter_delay = (self._win_len - 1) / 2.0
        return int(filter_delay)

    def _step(self, v_sample: float) -> float:
        # 1) Optional pre-filter (Moving Average)
        self._buffer.append(float(v_sample))

        # Warm-up until buffer fills (for win_len>1)
        if len(self._buffer) < self._win_len:
            self._prev_val = float(self._buffer[-1])
            self._total_samples += 1
            return self.NOMINAL_FREQ_HZ

        v_filt = float(sum(self._buffer) / len(self._buffer))

        # 2) Zero-crossing detection via sign change
        # We treat exact zeros carefully: np.sign(0)=0 can cause spurious transitions.
        curr_sign = 1.0 if v_filt > 0.0 else (-1.0 if v_filt < 0.0 else 0.0)
        prev_sign = (
            1.0 if self._prev_val > 0.0 else (-1.0 if self._prev_val < 0.0 else 0.0)
        )

        # Crossing condition: sign flips between + and -, ignoring zeros
        crossed = (curr_sign != 0.0) and (prev_sign != 0.0) and (curr_sign != prev_sign)

        if crossed and (self._total_samples > 0):
            self._handle_crossing(v_curr=v_filt)

        # update
        self._prev_val = v_filt
        self._total_samples += 1

        return float(self._f_est)

    def _handle_crossing(self, v_curr: float) -> None:
        """
        Linear interpolation to estimate fractional crossing time between samples.

        We approximate crossing time as:
          delta = |v_prev| / (|v_prev| + |v_curr|)
        with crossing at sample index (k-1)+delta, where k = self._total_samples.

        Then estimate half-cycle duration from consecutive crossings.
        """
        abs_prev = abs(self._prev_val)
        abs_curr = abs(v_curr)
        denom = abs_prev + abs_curr

        if denom > self.EPSILON:
            delta = abs_prev / denom
        else:
            # Degenerate case: both are ~0; assume mid-point
            delta = 0.5

        # Current crossing absolute index (fractional)
        # self._total_samples is the index of "current sample" in this step BEFORE increment.
        # v_curr corresponds to sample index self._total_samples (k),
        # and v_prev corresponds to (k-1).
        current_cross_absolute = float((self._total_samples - 1) + delta)

        if self._first_cross_detected:
            samples_diff = current_cross_absolute - self._last_cross_absolute

            if samples_diff > self.EPSILON:
                t_half_cycle = samples_diff / self._fs
                new_f = 1.0 / (2.0 * t_half_cycle)

                # Sanity check bounds
                if self.MIN_VALID_FREQ_HZ < new_f < self.MAX_VALID_FREQ_HZ:
                    self._f_est = float(new_f)

        self._last_cross_absolute = current_cross_absolute
        self._first_cross_detected = True
