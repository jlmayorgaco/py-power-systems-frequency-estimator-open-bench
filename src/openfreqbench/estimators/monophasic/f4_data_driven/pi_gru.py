"""
estimators/monophasic/f4_data_driven/pi_gru.py

PIGRUEstimator — Physics-Informed Gated Recurrent Unit frequency estimator.
Ported from legacy_sgsma/pigru_model.py
"""

from __future__ import annotations

import warnings
from typing import Any
from collections import deque
import numpy as np

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)

class PIGRUEstimator(BaseEstimator):
    """
    Physics-Informed GRU (PI-GRU) frequency estimator.
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
    def default_config(cls) -> dict[str, Any]:
        return {
            "fs": 10_000.0,
            "window_size": 256,
            "hidden_dim": 128,
            "num_layers": 2,
            "weights_path": "legacy_sgsma/pi_gru_pmu.pt",
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size", default=256, type="int",
                    values=[128, 256, 512], description="Input context window (samples)."
                ),
                TuningParam(
                    name="hidden_dim", default=128, type="int",
                    values=[32, 64, 128, 256], description="GRU hidden state dimension H."
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ
        self.window_size = int(self._config.get("window_size", 256))
        self.buffer = deque(maxlen=self.window_size)
        
        self.is_valid = False
        try:
            import torch
            import torch.nn as nn
            class AttentionBlock(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                        nn.Linear(hidden_dim // 2, 1), nn.Softmax(dim=1)
                    )
                def forward(self, x):
                    weights = self.attention(x)
                    context = torch.sum(weights * x, dim=1)
                    return context, weights
            class PIDRE_Model(nn.Module):
                def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv1d(input_dim, 32, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(32)
                    )
                    self.gru = nn.GRU(32, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
                    self.attn = AttentionBlock(hidden_dim)
                    self.fc_freq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
                def forward(self, x):
                    x_conv = x.permute(0, 2, 1)
                    feat = self.conv(x_conv)
                    feat = feat.permute(0, 2, 1)
                    out, _ = self.gru(feat)
                    context, _ = self.attn(out)
                    delta_freq = self.fc_freq(context).squeeze(-1)
                    return delta_freq + 60.0

            import os
            self.model = PIDRE_Model(
                hidden_dim=int(self._config.get("hidden_dim", 128)),
                num_layers=int(self._config.get("num_layers", 2))
            )
            # Try loading weights safely
            wp = self._config.get("weights_path", "")
            if os.path.exists(wp):
                self.model.load_state_dict(torch.load(wp, map_location="cpu"))
                self.model.eval()
                self.is_valid = True
            else:
                warnings.warn(f"PI-GRU weights not found at {wp}. Running in STUB mode.")
                
        except ImportError:
            warnings.warn("PyTorch not installed. PI-GRU estimator running in STUB mode.")

    def structural_latency_samples(self) -> int:
        return self.window_size

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        self.buffer.append(z)
        
        if len(self.buffer) < self.window_size or not self.is_valid:
            return EstimatorOutput(frequency_hz=self._f_est, valid=False)

        import torch
        with torch.no_grad():
            x = torch.tensor(list(self.buffer), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            self._f_est = self.model(x).item()
            
        return EstimatorOutput(frequency_hz=self._f_est, valid=True)

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
