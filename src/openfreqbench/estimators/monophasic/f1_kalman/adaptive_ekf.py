"""
estimators/monophasic/f1_kalman/adaptive_ekf.py  [SCAFFOLD]

AdaptiveEKFEstimator — EKF with innovation-based gain adaptation (no Huber).

Algorithm sketch
─────────────────
  A simpler adaptive variant than RAEKF:
  - Measures innovation covariance empirically over a sliding window.
  - Adjusts R̂ accordingly without the Huber M-weighting step.
  - Suitable for smoothly varying noise levels (not impulsive).
"""
from __future__ import annotations

from typing import Any, Dict

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class AdaptiveEKFEstimator(BaseEstimator):
    """Adaptive EKF (innovation-based R adaptation, no Huber) — scaffold."""

    SPEC = EstimatorSpec(
        name="Adaptive_EKF",
        family="Kalman",
        family_path="monophasic/f1_kalman",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "q_omega": 1.0, "r": 0.01,
                "window_size": 50, "forget_factor": 0.98}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="q_omega", default=1.0, type="float",
                            range=(1e-4, 10.0, 8), description="Process noise (ω)."),
                TuningParam(name="r", default=0.01, type="float",
                            range=(1e-4, 1.0, 8), description="Initial measurement noise."),
                TuningParam(name="forget_factor", default=0.98, type="float",
                            values=[0.95, 0.97, 0.98, 0.99, 0.995],
                            description="Exponential forgetting for R adaptation."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        fs = float(self._config.get("fs", 10_000.0))
        return int(2 * fs / self.NOMINAL_FREQ_HZ)

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        raise NotImplementedError("To be implemented in next phase")

    def _step(self, v_sample: float) -> float:
        raise NotImplementedError("To be implemented in next phase")
