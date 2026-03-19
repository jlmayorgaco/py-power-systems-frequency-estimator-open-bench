"""
estimators/monophasic/f4_data_driven/tcn_frequency.py  [SCAFFOLD]

TCNFrequencyEstimator — Temporal Convolutional Network with dilated causal convolutions.

Algorithm sketch
─────────────────
  A TCN stacks residual blocks of dilated causal convolutions with
  exponentially increasing dilation factors (1, 2, 4, ..., 2^{n_levels-1}).
  The receptive field grows as:
    RF = (kernel_size − 1) · 2 · (2^n_levels − 1) + 1
  enabling long-range temporal dependencies with causal constraints.

  Each residual block:
    x → DilatedCausalConv1D → WeightNorm → ReLU → Dropout
      → DilatedCausalConv1D → WeightNorm → ReLU → Dropout
    + 1×1 Conv (residual connection if channels differ)

References
──────────
  Bai, S., Kolter, J.Z., & Koltun, V. (2018). "An Empirical Evaluation
  of Generic Convolutional and Recurrent Networks for Sequence Modeling."
  arXiv:1803.01271.
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


class TCNFrequencyEstimator(BaseMLEstimator):
    """TCN with dilated causal convolutions — scaffold (not yet implemented)."""

    SPEC = EstimatorSpec(
        name="TCN_Freq",
        family="DataDriven",
        family_path="monophasic/f4_data_driven",
        complexity="O(window_size · n_channels · kernel_size · n_levels)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "fs": 10_000.0,
            "window_size": 128,
            "n_channels": 32,
            "n_levels": 4,
            "kernel_size": 3,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="window_size", default=128, type="int",
                            values=[64, 128, 256],
                            description="Input context window length (samples)."),
                TuningParam(name="n_channels", default=32, type="int",
                            values=[16, 32, 64, 128],
                            description="Number of channels in each residual block."),
                TuningParam(name="n_levels", default=4, type="int",
                            values=[2, 3, 4, 5, 6],
                            description="Number of dilated residual blocks."),
                TuningParam(name="kernel_size", default=3, type="int",
                            values=[2, 3, 4, 5],
                            description="Convolutional kernel size within each block."),
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
        return int(self._config.get("window_size", 128))

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        raise NotImplementedError("To be implemented in next phase")
