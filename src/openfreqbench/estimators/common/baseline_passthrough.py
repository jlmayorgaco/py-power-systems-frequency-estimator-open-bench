"""
estimators/common/baseline_passthrough.py  [CANONICAL]

BaselinePassthrough — returns the nominal frequency every sample.

This is the error floor for all benchmarks.  An estimator that cannot beat
the passthrough on a given scenario × metric pair is not useful.
"""
from __future__ import annotations

from typing import Any, Dict

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningSpec,
)


class BaselinePassthrough(BaseEstimator):
    """
    Trivial baseline: always outputs ``f_nom`` regardless of the input signal.

    Serves as the lower bound on benchmark accuracy.  Any real estimator must
    outperform this on well-conditioned signals or it provides no value.
    """

    SPEC = EstimatorSpec(
        name="Baseline_Passthrough",
        family="Baseline",
        family_path="common",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "f_nom": 60.0}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(params=[], objective="RMSE_HZ")

    def reset(self) -> None:
        self._f_nom: float = float(self._config.get("f_nom", 60.0))

    def structural_latency_samples(self) -> int:
        return 0

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        return EstimatorOutput(frequency_hz=self._f_nom, valid=True)

    # Backward-compat: old code calling _step() still works
    def _step(self, v_sample: float) -> float:
        return self._f_nom
