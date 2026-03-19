"""
estimators/monophasic/f4_data_driven/temporal_cnn.py  [SCAFFOLD]

TemporalCNNEstimator — 1D Convolutional Neural Network for frequency regression.

Algorithm sketch
─────────────────
  A stack of 1D causal convolutional layers with ReLU activations
  processes a fixed-length window of raw voltage samples.  Global
  average pooling (or the final timestep) reduces the feature map to
  a vector, which is then projected to a scalar frequency estimate.

  Architecture:
    Input:  [batch, window_size, 1]
    Conv1D(n_filters, kernel_size, padding='causal') × n_layers
    GlobalAvgPool → Linear → f_hat (Hz)

References
──────────
  LeCun, Y., et al. (1998). "Gradient-based learning applied to document
  recognition." Proc. IEEE, 86(11), 2278–2324.
"""
from __future__ import annotations

from typing import Any, Dict

from openfreqbench.estimators.common.ml_base import BaseMLEstimator
from openfreqbench.estimators.common.ml_types import (
    DataProtocolSpec,
    ModelSpec,
    TrainingSpec,
)
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class TemporalCNNEstimator(BaseMLEstimator):
    """1D Temporal CNN frequency regressor — scaffold (not yet implemented)."""

    SPEC = EstimatorSpec(
        name="TemporalCNN",
        family="DataDriven",
        family_path="monophasic/f4_data_driven",
        complexity="O(window_size · n_filters · kernel_size · n_layers)",
        latency_type="block",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "fs": 10_000.0,
            "window_size": 256,
            "n_filters": 32,
            "kernel_size": 3,
            "n_layers": 4,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="window_size", default=256, type="int",
                            values=[128, 256, 512],
                            description="Input window length (samples)."),
                TuningParam(name="n_filters", default=32, type="int",
                            values=[16, 32, 64, 128],
                            description="Number of convolutional filters per layer."),
                TuningParam(name="kernel_size", default=3, type="int",
                            values=[3, 5, 7, 11],
                            description="Convolutional kernel size."),
                TuningParam(name="n_layers", default=4, type="int",
                            values=[2, 3, 4, 6],
                            description="Number of convolutional layers."),
            ],
            objective="RMSE_HZ",
        )

    @classmethod
    def model_spec(cls) -> ModelSpec:
        raise NotImplementedError("To be implemented in next phase")

    @classmethod
    def training_spec(cls) -> TrainingSpec:
        raise NotImplementedError("To be implemented in next phase")

    @classmethod
    def data_protocol_spec(cls) -> DataProtocolSpec:
        raise NotImplementedError("To be implemented in next phase")

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 256))

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        raise NotImplementedError("To be implemented in next phase")
