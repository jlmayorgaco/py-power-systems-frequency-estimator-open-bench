"""
estimators/monophasic/f5_parametric/ipdft.py

IpDFT (Interpolated Discrete Fourier Transform)
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

class IpDFTEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="IpDFT",
        family="Parametric",
        family_path="monophasic/f5_parametric",
        complexity="O(N log N)",
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
            "cycles": 3,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="cycles", default=3, type="int", values=[1, 2, 3, 5, 10], description="Cycles used in rolling IpDFT window."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        cycles = int(self._config.get("cycles", 3))
        self.sz = max(1, int((fs / self.NOMINAL_FREQ_HZ) * cycles))
        self.buf = deque(maxlen=self.sz)
        self.win = np.hanning(self.sz)
        self.res = fs / self.sz

    def structural_latency_samples(self) -> int:
        return self.sz // 2

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        self.buf.append(z)
        
        if len(self.buf) < self.sz:
            return EstimatorOutput(frequency_hz=self.NOMINAL_FREQ_HZ, valid=False)
            
        sp_c = np.fft.rfft(np.array(self.buf) * self.win)
        sp = np.abs(sp_c)
        k = int(np.argmax(sp))
        
        if k == 0 or k == len(sp) - 1:
            f_out = k * self.res
        else:
            denom = 2.0 * sp_c[k] - sp_c[k - 1] - sp_c[k + 1]
            if abs(denom) < 1e-10:
                f_out = k * self.res
            else:
                delta = float(np.real((sp_c[k + 1] - sp_c[k - 1]) / denom))
                delta = float(np.clip(delta, -0.5, 0.5))
                f_out = (k + delta) * self.res
                
        f_out = float(np.clip(f_out, 40.0, 80.0))
        return EstimatorOutput(frequency_hz=f_out, valid=True)

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
