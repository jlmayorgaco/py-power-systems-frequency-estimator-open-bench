"""
estimators/monophasic/f1_kalman/ukf.py  [STUB]

UKFEstimator — Unscented Kalman Filter for frequency estimation.

Algorithm sketch
────────────────
  1. Nonlinear sinusoidal state model: x = [A·cos φ, ω],  φ propagated
     separately.
  2. Sigma points chosen via the symmetric unscented transform (alpha, β, κ).
  3. Propagate sigma points through the nonlinear measurement function
     h(x) = A·sin(φ + ω·Ts) to form predicted mean and covariance.
  4. Standard Kalman update on the predicted/actual innovation.

Status: STUB — outputs nominal frequency with valid=False until implemented.
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


class UKFEstimator(BaseEstimator):
    """
    Unscented Kalman Filter (UKF) frequency estimator — stub.

    Not yet implemented; registers in the estimator catalogue so benchmarks
    can enumerate it.  All outputs carry ``valid=False``.
    """

    SPEC = EstimatorSpec(
        name="UKF",
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
        return {
            "fs": 10_000.0,
            "q_omega": 1.0,
            "r": 0.01,
            "alpha": 1e-3,
            "beta": 2.0,
            "kappa": 0.0,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="q_omega",
                    default=1.0,
                    type="float",
                    range=(1e-4, 10.0, 8),
                    description="Process noise for angular frequency state.",
                ),
                TuningParam(
                    name="r",
                    default=0.01,
                    type="float",
                    range=(1e-4, 1.0, 8),
                    description="Measurement noise variance.",
                ),
                TuningParam(
                    name="alpha",
                    default=1e-3,
                    type="float",
                    values=[1e-4, 1e-3, 1e-2, 0.1, 1.0],
                    description="UKF spread parameter alpha.",
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return 0

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        return EstimatorOutput(frequency_hz=self._f_est, valid=False)

    def _step(self, v_sample: float) -> float:
        return self._f_est
