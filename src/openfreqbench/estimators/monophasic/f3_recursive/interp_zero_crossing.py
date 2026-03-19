"""
estimators/monophasic/f3_recursive/interp_zero_crossing.py  [SCAFFOLD]

InterpolatedZeroCrossingEstimator — sub-sample interpolated zero-crossing.

Algorithm sketch
─────────────────
  Detects sign changes and fits a linear or parabolic interpolant
  between the two bracketing samples to estimate the exact fractional
  zero-crossing time, giving sub-sample frequency resolution.
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


class InterpolatedZeroCrossingEstimator(BaseEstimator):
    """Sub-sample interpolated zero-crossing estimator — scaffold."""

    SPEC = EstimatorSpec(
        name="IZC",
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
        return {"fs": 10_000.0, "interpolation": "linear", "filter_win": 5}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="filter_win", default=5, type="int",
                            values=[1, 3, 5, 7, 11],
                            description="Output smoothing window (samples)."),
                TuningParam(name="interpolation", default="linear", type="categorical",
                            values=["linear", "parabolic"],
                            description="Interpolation order at zero crossing."),
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
