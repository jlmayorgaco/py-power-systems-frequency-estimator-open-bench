"""
estimators/monophasic/f3_recursive/vff_rls.py

Variable Forgetting Factor RLS (VFF-RLS)
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

class VFFRLSEstimator(BaseEstimator):
    SPEC = EstimatorSpec(
        name="VFF_RLS",
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
            "lam_min": 0.98,
            "lam_max": 0.9995,
            "ka": 3.0,
            "decim": 50,
            "win_smooth": 20,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(name="lam_min", default=0.98, type="float", values=[0.95, 0.98, 0.99], description="VFF lower bound."),
                TuningParam(name="ka", default=3.0, type="float", values=[1.0, 3.0, 5.0, 10.0], description="VFF dynamic response gain."),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self.lam_min = float(self._config.get("lam_min", 0.98))
        self.lam_max = float(self._config.get("lam_max", 0.9995))
        self.ka = float(self._config.get("ka", 3.0))
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

        self._e_pow = 0.0
        self._v_pow = 0.01

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

        alpha_f = 1.0 / self.ka
        self._e_pow = (1 - alpha_f) * self._e_pow + alpha_f * (e**2)
        self._v_pow = (1 - alpha_f) * self._v_pow + alpha_f * (d**2)
        
        ratio = min(1.0, self._e_pow / (self._v_pow + 1e-9))
        lam = self.lam_max - (self.lam_max - self.lam_min) * ratio
        lam = np.clip(lam, self.lam_min, self.lam_max)

        Pphi = self.P @ phi
        denom = lam + float(phi @ Pphi)
        if denom <= 0.0: denom = 1e-9
        
        K = Pphi / denom
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Pphi)) / lam
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
