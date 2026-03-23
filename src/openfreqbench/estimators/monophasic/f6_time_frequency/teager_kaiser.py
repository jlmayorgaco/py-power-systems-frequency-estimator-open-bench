"""
estimators/monophasic/f6_time_frequency/teager_kaiser.py

Teager-Kaiser Energy Operator (DESA-2 Algorithm)
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

class TeagerKaiserEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="TKEO_DESA2",
        family="Time-Frequency",
        family_path="monophasic/f6_time_frequency",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
        is_three_phase=False,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "fs": 10000.0,
            "smooth_win": 167,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="smooth_win", default=167, type="int", values=[16, 83, 167, 334], description="Smoothing MA filter length (samples)"),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self._dt = 1.0 / fs
        self.win = int(self._config.get("smooth_win", 167))
        self.buf = deque(maxlen=5)
        self.f_buf = deque(maxlen=self.win)
        self.last_valid_f = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return self.win + 2

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        self.buf.append(z)
        
        if len(self.buf) < 5:
            return EstimatorOutput(frequency_hz=self.NOMINAL_FREQ_HZ, valid=False)

        x_n = self.buf[2]
        x_nm1 = self.buf[1]
        x_np1 = self.buf[3]
        x_nm2 = self.buf[0]
        x_np2 = self.buf[4]
        
        psi_x = x_n ** 2 - x_nm1 * x_np1
        y_n = x_np1 - x_nm1
        y_nm1 = x_n - x_nm2
        y_np1 = x_np2 - x_n
        psi_y = y_n ** 2 - y_nm1 * y_np1
        
        if psi_x <= 1e-4:
            f = self.last_valid_f
        else:
            val = 1.0 - psi_y / (2.0 * psi_x)
            if abs(val) > 1.0:
                val = math.copysign(1.0, val)
            w = 0.5 * math.acos(val)
            f = (w / self._dt) / (2 * math.pi)
            
        if f > self.MAX_VALID_FREQ_HZ or f < self.MIN_VALID_FREQ_HZ or math.isnan(f):
            f = self.last_valid_f

        if len(self.f_buf) > 0 and abs(f - self.f_buf[-1]) > 5.0:
            f = self.f_buf[-1]
        
        self.f_buf.append(f)
        self.last_valid_f = f
        
        is_valid = len(self.f_buf) >= self.win
        f_out = float(np.mean(self.f_buf)) if is_valid else self.NOMINAL_FREQ_HZ
        
        return EstimatorOutput(frequency_hz=f_out, valid=is_valid)

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
