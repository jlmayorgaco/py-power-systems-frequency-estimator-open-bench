"""
estimators/monophasic/f5_parametric/tft.py

Taylor-Fourier Transform Estimator (K=2)
Ported from legacy_sgsma/estimators.py
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from typing import Any

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)

class TFTEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="TFT",
        family="Parametric",
        family_path="monophasic/f5_parametric",
        complexity="O(N)",
        latency_type="symmetric",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
        is_three_phase=False,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "fs": 10000.0,
            "win_cycles": 1.0,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="win_cycles", default=1.0, type="float", values=[0.5, 1.0, 2.0, 3.0], description="Cycles used in rolling TFT window."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        win_cycles = float(self._config.get("win_cycles", 1.0))
        self.N = max(10, int((fs / self.NOMINAL_FREQ_HZ) * win_cycles))
        self.buf = deque(maxlen=self.N)
        
        t_vec = np.arange(self.N) * (1.0 / fs)
        t_vec = t_vec - np.mean(t_vec)
        w = 2 * math.pi * self.NOMINAL_FREQ_HZ
        
        H = np.zeros((self.N, 6))
        H[:, 0] = np.cos(w * t_vec)
        H[:, 1] = np.sin(w * t_vec)
        H[:, 2] = t_vec * np.cos(w * t_vec)
        H[:, 3] = t_vec * np.sin(w * t_vec)
        H[:, 4] = (t_vec ** 2) * np.cos(w * t_vec)
        H[:, 5] = (t_vec ** 2) * np.sin(w * t_vec)
        self.H_pinv = np.linalg.pinv(H)

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        self.buf.append(z)
        
        if len(self.buf) < self.N:
            return EstimatorOutput(frequency_hz=self.NOMINAL_FREQ_HZ, valid=False)
            
        y = np.array(self.buf)
        coeffs = self.H_pinv @ y
        a0, b0, a1, b1 = coeffs[0:4]
        
        num = b0 * a1 - a0 * b1
        den = a0 ** 2 + b0 ** 2
        
        if den < 1e-6:
            f_out = self.NOMINAL_FREQ_HZ
        else:
            df = (1.0 / (2 * math.pi)) * (num / den)
            f_out = self.NOMINAL_FREQ_HZ + df
            
        f_out = float(np.clip(f_out, 40.0, 80.0))
        return EstimatorOutput(frequency_hz=f_out, valid=True)

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
