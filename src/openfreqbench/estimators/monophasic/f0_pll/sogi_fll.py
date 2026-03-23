"""
estimators/monophasic/f0_pll/sogi_fll.py

SOGI-FLL (Second-Order Generalized Integrator Frequency-Locked Loop)
Ported from legacy_sgsma/estimators.py - Improved Discretization Algorithm.
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

class SOGIFLLEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="SOGI_FLL",
        family="PLL",
        family_path="monophasic/f0_pll",
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
            "k_gain": 1.414,
            "gamma": 50.0,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="k_gain", default=1.414, type="float", values=[0.707, 1.0, 1.414, 2.0], description="SOGI damping factor."),
                TuningParam(name="gamma", default=50.0, type="float", values=[10.0, 50.0, 100.0, 150.0], description="FLL adaptation gain."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self._dt = 1.0 / fs
        self.k = float(self._config.get("k_gain", 1.414))
        self.gamma = float(self._config.get("gamma", 50.0))
        
        self.w = 2.0 * math.pi * self.NOMINAL_FREQ_HZ
        self.v_alpha = 0.0
        self.v_beta = 0.0
        
        self.smooth_win = max(1, int(fs / self.NOMINAL_FREQ_HZ))
        self.f_buf = deque(maxlen=self.smooth_win)
        
        _half = max(1, int(fs / self.NOMINAL_FREQ_HZ / 2))
        self._mag_buf = deque([0.5], maxlen=_half)

    def structural_latency_samples(self) -> int:
        return self.smooth_win

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        e = z - self.v_alpha

        v_alpha_pred = self.v_alpha + self._dt * (self.w * e * self.k - self.w * self.v_beta)
        v_beta_pred  = self.v_beta  + self._dt * (self.w * self.v_alpha)

        e_pred = z - v_alpha_pred
        dot_alpha = 0.5 * ((self.w * e * self.k - self.w * self.v_beta) + 
                           (self.w * e_pred * self.k - self.w * v_beta_pred))
        dot_beta = 0.5 * (self.w * self.v_alpha + self.w * v_alpha_pred)

        self.v_alpha += self._dt * dot_alpha
        self.v_beta  += self._dt * dot_beta

        mag_sq = self.v_alpha ** 2 + self.v_beta ** 2
        self._mag_buf.append(mag_sq)
        mag_avg = float(np.mean(self._mag_buf))
        if mag_avg < 1e-4:
            mag_avg = 1e-4

        w_dot = -self.gamma * (z - self.v_alpha) * self.v_beta / mag_avg
        self.w += w_dot * self._dt
        self.w = float(np.clip(self.w, 2.0 * math.pi * 40.0, 2.0 * math.pi * 80.0))

        f_inst = self.w / (2.0 * math.pi)
        self.f_buf.append(f_inst)
        
        is_valid = len(self.f_buf) >= self.smooth_win
        f_out = float(np.mean(self.f_buf)) if is_valid else self.NOMINAL_FREQ_HZ
        
        return EstimatorOutput(
            frequency_hz=f_out,
            valid=is_valid,
            phase_rad=(timestamp * self.w) % (2*math.pi),
            amplitude_pu=math.sqrt(mag_avg)
        )

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
