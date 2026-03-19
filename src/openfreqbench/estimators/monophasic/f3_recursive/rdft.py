"""
estimators/monophasic/f3_recursive/rdft.py  [CANONICAL]

RDFTEstimator — Recursive (Sliding-Window) DFT frequency estimator.

Algorithm
─────────
  1. Maintain a complex phasor at the nominal frequency bin using the
     sliding-DFT identity (O(1) per sample):
       X[n] = (X[n-1] + v[n] − v[n-N]) · e^(j·2π·k₀/N)
     where k₀ = round(f_nom · N / fs).

  2. Extract frequency from the inter-sample phase difference:
       phase_diff[n] = arg(X[n] · conj(X[n-1]))
       f̂[n] = f_nom + mean(phase_diff, window=N_avg) · fs / (2π)

  3. Output is held at the nominal frequency until the sliding buffer is full.

Complexity:  O(1) per sample (O(N) amortised across the buffer lifetime).
Latency:     window_size samples (one full buffer fill).

References
──────────
  Jacobsen, E. & Lyons, R. (2003). "The sliding DFT." IEEE Signal Process.
  Mag., 20(2), 74–80.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class RDFTEstimator(BaseEstimator):
    """
    Recursive (Sliding) DFT frequency estimator.

    Tracks the fundamental phasor without a full FFT on every sample.
    Well-suited for near-nominal frequency tracking in real-time applications.
    """

    SPEC = EstimatorSpec(
        name="RDFT",
        family="Recursive",
        family_path="monophasic/f3_recursive",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "window_size": 256, "phase_avg": 8}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size",
                    default=256,
                    type="int",
                    values=[64, 128, 256, 512, 1024],
                    description=(
                        "Sliding DFT window length N (samples). "
                        "Larger → finer frequency resolution, higher latency."
                    ),
                ),
                TuningParam(
                    name="phase_avg",
                    default=8,
                    type="int",
                    values=[1, 2, 4, 8, 16, 32],
                    description=(
                        "Number of phase-difference samples to average for "
                        "the frequency estimate.  Larger → smoother, slower."
                    ),
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs        = float(self._config.get("fs",          10_000.0))
        N         = int(self._config.get("window_size",   256))
        avg       = int(self._config.get("phase_avg",     8))

        self._fs      = fs
        self._N       = N
        self._avg     = avg

        # Nominal DFT bin and rotation factor
        k0            = max(1, round(self.NOMINAL_FREQ_HZ * N / fs))
        self._k0      = k0
        angle         = 2.0 * math.pi * k0 / N
        self._W       = complex(math.cos(angle), math.sin(angle))

        # State
        self._buf:          deque = deque([0.0] * N, maxlen=N)
        self._X:            complex = 0j
        self._X_prev:       complex = 0j
        self._n:            int = 0
        self._f_est:        float = self.NOMINAL_FREQ_HZ
        self._phase_diffs:  deque = deque(maxlen=avg)

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 256))

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        f = self._step(float(voltage))
        valid = (
            self._n > self.structural_latency_samples()
            and self.MIN_VALID_FREQ_HZ <= f <= self.MAX_VALID_FREQ_HZ
        )
        return EstimatorOutput(frequency_hz=f, valid=valid)

    # ── Internal algorithm ────────────────────────────────────────────────────

    def _step(self, v_sample: float) -> float:
        v        = float(v_sample)
        oldest   = self._buf[0]
        self._buf.append(v)
        self._n += 1

        X_new = (self._X + v - oldest) * self._W

        if self._n > self._N:
            if abs(self._X_prev) > 1e-10 and abs(X_new) > 1e-10:
                pd         = X_new * self._X_prev.conjugate()
                phase_diff = math.atan2(pd.imag, pd.real)
                self._phase_diffs.append(phase_diff)

            if self._phase_diffs:
                avg_pd  = sum(self._phase_diffs) / len(self._phase_diffs)
                f_raw   = self.NOMINAL_FREQ_HZ + avg_pd * self._fs / (2.0 * math.pi)
                if self.MIN_VALID_FREQ_HZ < f_raw < self.MAX_VALID_FREQ_HZ:
                    self._f_est = f_raw

        self._X_prev = self._X
        self._X      = X_new
        return self._f_est
