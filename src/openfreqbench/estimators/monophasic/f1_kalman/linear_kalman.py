"""
estimators/monophasic/f1_kalman/linear_kalman.py  [SCAFFOLD]

LinearKalmanEstimator — Linear Kalman Filter via DFT state-space model.

Algorithm sketch
─────────────────
  State: [a_c, a_s, ω] (cosine/sine phasor + frequency).
  Measurement: v[n] = a_c·cos(ω₀·n·Ts) + a_s·sin(ω₀·n·Ts).
  Linearised around nominal ω₀ each step; frequency extracted from phasor rate.

References
──────────
  Brown, R.G. & Hwang, P.Y.C. (1997). Introduction to Random Signals
  and Applied Kalman Filtering.
"""

from __future__ import annotations

from typing import Any

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class LinearKalmanEstimator(BaseEstimator):
    """Linear Kalman Filter (phasor state-space) — scaffold."""

    SPEC = EstimatorSpec(
        name="LKF",
        family="Kalman",
        family_path="monophasic/f1_kalman",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"fs": 10_000.0, "q": 1.0, "r": 0.01}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="q",
                    default=1.0,
                    type="float",
                    range=(1e-4, 100.0, 8),
                    description="Process noise variance.",
                ),
                TuningParam(
                    name="r",
                    default=0.01,
                    type="float",
                    range=(1e-4, 1.0, 8),
                    description="Measurement noise variance.",
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        fs = float(self._config.get("fs", 10_000.0))
        return int(fs / self.NOMINAL_FREQ_HZ)

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        raise NotImplementedError("To be implemented in next phase")

    def _step(self, v_sample: float) -> float:
        raise NotImplementedError("To be implemented in next phase")
