"""
estimators/monophasic/f4_data_driven/pi_gru.py  [STUB]

PIGRUEstimator — Physics-Informed Gated Recurrent Unit frequency estimator.

Algorithm sketch
────────────────
  1. A compact GRU (2 layers, hidden_dim units) maps a sliding window of raw
     voltage samples directly to instantaneous frequency.
  2. Physics-informed regularisation: the training loss augments MSE with
     a sinusoidal reconstruction penalty |v[n] − Â·sin(2π·f̂·n/fs + φ̂)|²
     to prevent physically implausible frequency outputs.
  3. Inference is purely feed-forward after training; no online adaptation.
  4. Model weights are loaded from ``weights_path`` at construction time.
     If the file is absent the estimator falls back to valid=False.

Status: STUB — outputs nominal frequency with valid=False until implemented /
pretrained weights are provided.
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


class PIGRUEstimator(BaseEstimator):
    """
    Physics-Informed GRU (PI-GRU) frequency estimator — stub.

    Not yet implemented / weights not bundled; registers in the estimator
    catalogue so benchmarks can enumerate it.  All outputs carry ``valid=False``.
    """

    SPEC = EstimatorSpec(
        name="PI_GRU",
        family="Data-Driven",
        family_path="monophasic/f4_data_driven",
        complexity="O(H²)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "fs": 10_000.0,
            "window_size": 256,
            "hidden_dim": 64,
            "num_layers": 2,
            "weights_path": "",
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size",
                    default=256,
                    type="int",
                    values=[128, 256, 512],
                    description="Input context window (samples).",
                ),
                TuningParam(
                    name="hidden_dim",
                    default=64,
                    type="int",
                    values=[32, 64, 128, 256],
                    description="GRU hidden state dimension H.",
                ),
                TuningParam(
                    name="num_layers",
                    default=2,
                    type="int",
                    values=[1, 2, 3],
                    description="Number of stacked GRU layers.",
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 256))

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        return EstimatorOutput(frequency_hz=self._f_est, valid=False)

    def _step(self, v_sample: float) -> float:
        return self._f_est
