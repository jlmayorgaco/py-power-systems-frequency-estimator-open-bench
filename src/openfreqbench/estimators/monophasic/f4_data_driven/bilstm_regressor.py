"""
estimators/monophasic/f4_data_driven/bilstm_regressor.py  [SCAFFOLD]

BiLSTMRegressorEstimator — Bidirectional LSTM network for frequency regression.

Algorithm sketch
─────────────────
  A bidirectional LSTM processes the input window in both forward and
  backward directions simultaneously.  The concatenated hidden states
  from both directions at the centre timestep are projected by a linear
  head to a scalar frequency estimate in Hz.

  Non-causal: requires the full window before producing a single
  centre-point estimate → structural latency = window_size // 2.

  Forward pass (per window):
    h_fwd = LSTM_forward(v[0..N-1])
    h_bwd = LSTM_backward(v[N-1..0])
    f_hat = Linear(concat(h_fwd[N//2], h_bwd[N//2]))

References
──────────
  Schuster, M. & Paliwal, K.K. (1997). "Bidirectional recurrent neural
  networks." IEEE Trans. Signal Processing, 45(11), 2673–2681.
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


class BiLSTMRegressorEstimator(BaseMLEstimator):
    """Bidirectional LSTM frequency regressor — scaffold (not yet implemented)."""

    SPEC = EstimatorSpec(
        name="BiLSTM_Regressor",
        family="DataDriven",
        family_path="monophasic/f4_data_driven",
        complexity="O(window_size · hidden_dim²)",
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
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": True,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="window_size", default=256, type="int",
                            values=[128, 256, 512],
                            description="Input sequence length (samples)."),
                TuningParam(name="hidden_dim", default=64, type="int",
                            values=[32, 64, 128, 256],
                            description="LSTM hidden state dimension (per direction)."),
                TuningParam(name="num_layers", default=2, type="int",
                            values=[1, 2, 3],
                            description="Number of stacked BiLSTM layers."),
                TuningParam(name="dropout", default=0.1, type="float",
                            values=[0.0, 0.1, 0.2, 0.3],
                            description="Dropout probability between BiLSTM layers."),
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
        return int(self._config.get("window_size", 256)) // 2

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        raise NotImplementedError("To be implemented in next phase")
