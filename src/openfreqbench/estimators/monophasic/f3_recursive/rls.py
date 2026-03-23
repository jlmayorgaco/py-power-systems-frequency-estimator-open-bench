"""
estimators/monophasic/f3_recursive/rls.py

Recursive Least Squares (RLS) AR(2) Decimated Estimator
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

class RLSEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="RLS",
        family="Recursive",
        family_path="monophasic/f3_recursive",
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
            "lam": 0.99,
            "win_smooth": 20,
            "decim": 50,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="lam", default=0.99, type="float", values=[0.98, 0.99, 0.995, 0.999], description="Forgetting factor lambda."),
                TuningParam(name="win_smooth", default=20, type="int", values=[5, 20, 50], description="Smoothing MA window size."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self.lam = float(self._config.get("lam", 0.99))
        self.decim = int(self._config.get("decim", 50))
        if self.decim < 1: self.decim = 1
        
        self._dt = 1.0 / fs
        self.DT_eff = self.decim * self._dt
        
        w0 = 2.0 * math.pi * self.NOMINAL_FREQ_HZ
        a1_60 = 2.0 * math.cos(w0 * self.DT_eff)
        self.theta = np.array([a1_60, -1.0], dtype=float)
        
        self.P = np.eye(2) * 10.0
        self.y_buf = deque([0.0, 0.0], maxlen=2)
        
        self._cnt = 0
        self.smooth_win = int(self._config.get("win_smooth", 20))
        self.f_buf = deque(maxlen=self.smooth_win)
        self._last_f = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return self.smooth_win * self.decim

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])
        self._cnt += 1
        if self._cnt < self.decim:
            return EstimatorOutput(frequency_hz=self._last_f, valid=len(self.f_buf) >= self.smooth_win)
        self._cnt = 0

        self.y_buf.append(z)
        if len(self.y_buf) < 2:
            return EstimatorOutput(frequency_hz=self._last_f, valid=False)

        phi_raw = np.array([self.y_buf[1], self.y_buf[0]], dtype=float)
        norm_phi = np.linalg.norm(phi_raw) + 1e-9
        phi = phi_raw / norm_phi
        d = z / norm_phi

        y_pred = float(self.theta @ phi)
        e = d - y_pred

        Pphi = self.P @ phi
        denom = self.lam + float(phi @ Pphi)
        if denom <= 0.0: denom = 1e-9
        
        K = Pphi / denom
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Pphi)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)

        a1 = float(np.clip(self.theta[0], -1.9999, 1.9999))
        val = float(np.clip(a1 / 2.0, -0.9999, 0.9999))
        
        try:
            w = math.acos(val) / self.DT_eff
            f_inst = w / (2.0 * math.pi)
        except ValueError:
            f_inst = self.NOMINAL_FREQ_HZ
            
        f_inst = float(np.clip(f_inst, 40.0, 80.0)) if np.isfinite(f_inst) else self.NOMINAL_FREQ_HZ
        self.f_buf.append(f_inst)
        self._last_f = float(np.mean(self.f_buf)) if len(self.f_buf) >= self.smooth_win else self.NOMINAL_FREQ_HZ

        return EstimatorOutput(frequency_hz=self._last_f, valid=len(self.f_buf) >= self.smooth_win)

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
