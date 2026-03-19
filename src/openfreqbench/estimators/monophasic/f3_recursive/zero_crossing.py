"""
estimators/monophasic/f3_recursive/zero_crossing.py  [CANONICAL]

ZeroCrossingEstimator — time-domain zero-crossing with moving-average
pre-filter and sub-sample linear interpolation.

Algorithm
─────────
  1. Apply a length-W moving-average filter to suppress noise.
  2. Detect sign changes in the filtered signal (positive zero crossings).
  3. Sub-sample linear interpolation between adjacent samples for the
     crossing location.
  4. Estimate frequency from the half-period between consecutive crossings:
       f = fs / (2 * (t_cross_k − t_cross_{k−1}))
  5. Output is held constant between crossings (zero-order hold).

Latency:    (W−1)/2 + fs/(4·f_nom) samples
Complexity: O(W) per sample
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class ZeroCrossingEstimator(BaseEstimator):
    """
    Time-domain zero-crossing frequency estimator.

    Simple, low-latency, and interpretable.  Performance degrades on
    heavily distorted or low-amplitude signals.
    """

    SPEC = EstimatorSpec(
        name="ZeroCrossing",
        family="TimeDomain",
        family_path="monophasic/f3_recursive",
        complexity="O(W)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    _EPSILON: float = 1e-9

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "filter_win": 5}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="filter_win",
                    default=5,
                    type="int",
                    values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            12, 15, 20, 25, 35, 50, 100, 120, 250],
                    description=(
                        "Moving-average pre-filter window (samples). "
                        "Larger → more noise rejection, higher latency and "
                        "slower transient response."
                    ),
                )
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._fs:              float = float(self._config.get("fs", 10_000.0))
        self._win_len:         int   = int(self._config.get("filter_win", 5))
        self._buffer:          deque = deque(maxlen=self._win_len)
        self._prev_val:        float = 0.0
        self._total_samples:   int   = 0
        self._last_cross:      float = 0.0
        self._first_cross:     bool  = False
        self._f_est:           float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        fs      = float(self._config.get("fs", 10_000.0))
        win_len = int(self._config.get("filter_win", 5))
        return int((win_len - 1) / 2.0 + fs / (4.0 * self.NOMINAL_FREQ_HZ))

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        f     = self._step(float(voltage))
        valid = (
            self._first_cross
            and self._total_samples > self.structural_latency_samples()
            and self.MIN_VALID_FREQ_HZ <= f <= self.MAX_VALID_FREQ_HZ
        )
        return EstimatorOutput(frequency_hz=f, valid=valid)

    # ── Internal algorithm ────────────────────────────────────────────────────

    def _step(self, v_sample: float) -> float:
        """Pure math — one voltage sample → frequency estimate (Hz)."""
        self._buffer.append(v_sample)
        if len(self._buffer) < self._win_len:
            return self.NOMINAL_FREQ_HZ

        v_filt    = sum(self._buffer) / len(self._buffer)
        curr_sign = 1 if v_filt >= 0 else -1
        prev_sign = 1 if self._prev_val >= 0 else -1

        if curr_sign != prev_sign and self._total_samples > 0:
            self._handle_crossing(v_filt)

        self._prev_val       = v_filt
        self._total_samples += 1
        return self._f_est

    def _handle_crossing(self, v_curr: float) -> None:
        abs_prev = abs(self._prev_val)
        abs_curr = abs(v_curr)
        denom    = abs_prev + abs_curr
        delta    = abs_prev / denom if denom > self._EPSILON else 0.5
        cross    = (self._total_samples - 1) + delta

        if self._first_cross:
            diff = cross - self._last_cross
            if diff > self._EPSILON:
                new_f = self._fs / (2.0 * diff)
                if self.MIN_VALID_FREQ_HZ < new_f < self.MAX_VALID_FREQ_HZ:
                    self._f_est = new_f

        self._last_cross  = cross
        self._first_cross = True
