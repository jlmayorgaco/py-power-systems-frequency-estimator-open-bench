"""
estimators/monophasic/f1_kalman/ra_ekf.py

RA_EKF — RoCoF-Augmented Extended Kalman Filter.
Ported from legacy_sgsma/ekf2.py.
"""

from __future__ import annotations

import math
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

class RAEKFFrequencyEstimator(BaseEstimator):
    """
    RA-EKF: Extended Kalman Filter augmented with ROCOF and event-gating
    for dynamic protection.

    State:
        x = [theta (phase), omega (freq), A (amplitude), domega (ROCOF)]^T
    """

    SPEC = EstimatorSpec(
        name="RA_EKF",
        family="Kalman",
        family_path="monophasic/f1_kalman",
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
            "q_param": 10.0,
            "r_param": 0.01,
            "inn_ref": 0.5,
            "event_thresh": 2.0,
            "fast_horizon_ms": 80.0,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="q_param", default=10.0, type="float", 
                    values=[0.1, 1.0, 10.0, 100.0, 1000.0],
                    description="Model noise baseline param"
                ),
                TuningParam(
                    name="r_param", default=0.01, type="float", 
                    values=[0.001, 0.01, 0.1, 1.0],
                    description="Measurement noise baseline param"
                ),
                TuningParam(
                    name="inn_ref", default=0.3, type="float", 
                    values=[0.1, 0.3, 0.5],
                    description="Nominal innovation scale"
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self._dt = 1.0 / fs
        
        q_param = float(self._config.get("q_param", 10.0))
        r_param = float(self._config.get("r_param", 0.01))
        self.inn_ref = float(self._config.get("inn_ref", 0.5))
        self.event_thresh = float(self._config.get("event_thresh", 2.0))
        self.fast_horizon = int((float(self._config.get("fast_horizon_ms", 80.0)) * 1e-3) * fs)

        self.x = np.array([0.0, 2 * np.pi * self.NOMINAL_FREQ_HZ, 1.0, 0.0], dtype=float)
        self.P = np.diag([1e-2, (2 * np.pi * 1.0)**2, 1e-2, (2 * np.pi * 5.0)**2])

        self.Q_slow = np.diag([1e-7, q_param, 1e-4, 10.0 * q_param])
        self.Q_fast = np.diag([1e-6, 50.0 * q_param, 1e-3, 500.0 * q_param])
        self.R_base = np.array([[r_param]])

        self.Q = self.Q_slow.copy()
        self.R = self.R_base.copy()

        self.fast_timer = 0
        self.inn_buf = deque(maxlen=200)
        self.I = np.eye(4)
        self.is_init = False
        self.A_init_est = 1.0

    def structural_latency_samples(self) -> int:
        return 0

    def _adaptive_QR(self, inn: float) -> None:
        abs_inn = abs(inn)
        ratio = np.clip(abs_inn / (self.inn_ref + 1e-8), 0.25, 4.0)
        q_scale_matrix = np.diag([ratio, ratio, ratio, 1.0])
        
        if self.fast_timer > 0:
            Q_temp = self.Q_fast.copy()
            Q_temp[3, 3] *= 0.1
            self.Q = Q_temp @ q_scale_matrix
        else:
            self.Q = self.Q_slow @ q_scale_matrix
        self.R = self.R_base * (1.0 / ratio)

    def _maybe_trigger_event(self, inn: float, z: float) -> None:
        abs_inn = abs(inn)
        self.inn_buf.append(inn)
        if len(self.inn_buf) < 10: return

        med_abs = np.median(np.abs(self.inn_buf))
        ref = max(self.inn_ref, med_abs)

        if abs_inn > self.event_thresh * ref:
            self.fast_timer = self.fast_horizon
            A_eff = max(self.x[2], 0.1)
            arg = np.clip(z / A_eff, -0.99, 0.99)
            self.x[0] = np.arcsin(arg)
            self.P[0, 0] = np.clip(self.P[0, 0] * 3.0, 0.0, 1.0)
            self.P[1, 1] = np.clip(self.P[1, 1], 0.0, (2 * np.pi * 5) ** 2)
            self.P[0, 1] = 0.0
            self.P[1, 0] = 0.0

    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        z = float(np.atleast_1d(voltage)[0])

        if not self.is_init:
            self.A_init_est = max(self.A_init_est * 0.95 + abs(z) * 0.05 * 1.41, 0.1)
            A_safe = max(self.A_init_est, 1.0)
            safe_z = np.clip(z / A_safe, -0.99, 0.99)
            self.x[0] = np.arcsin(safe_z)
            self.x[2] = self.A_init_est
            self.is_init = True
            return EstimatorOutput(frequency_hz=self.NOMINAL_FREQ_HZ, valid=True, phase_rad=self.x[0], amplitude_pu=self.x[2])

        if self.fast_timer > 0:
            self.fast_timer -= 1

        theta, omega, A, domega = self.x
        theta_pred = theta + omega * self._dt + 0.5 * domega * self._dt**2
        omega_pred = omega + domega * self._dt

        self.x = np.array([theta_pred, omega_pred, A, domega])

        F = np.eye(4)
        F[0, 1] = self._dt
        F[0, 3] = 0.5 * self._dt**2
        F[1, 3] = self._dt

        self.P = F @ self.P @ F.T + self.Q

        theta, omega, A, domega = self.x
        y_pred = A * np.sin(theta)
        inn = z - y_pred

        self._adaptive_QR(inn)
        self._maybe_trigger_event(inn, z)

        H = np.array([[A * np.cos(theta), 0.0, np.sin(theta), 0.0]])
        S = H @ self.P @ H.T + self.R
        inv_S = 1.0 / (S[0, 0] + 1e-12)
        K = self.P @ H.T * inv_S

        self.x += (K * inn).flatten()
        self.P = (self.I - K @ H) @ self.P

        omega_pre_clip = self.x[1]
        self.x[1] = np.clip(self.x[1], 2 * np.pi * 45.0, 2 * np.pi * 75.0)
        if self.x[1] != omega_pre_clip:
            self.P[1, 1] = min(self.P[1, 1], (2 * np.pi * 1.0) ** 2)
            
        self.x[3] = np.clip(self.x[3], -2 * np.pi * 15.0, 2 * np.pi * 15.0)
        self.x[0] %= 2 * np.pi
        self.x[2] = max(0.1, self.x[2])

        return EstimatorOutput(
            frequency_hz=self.x[1] / (2 * np.pi),
            valid=True,
            phase_rad=self.x[0],
            amplitude_pu=self.x[2],
        )

    def _step(self, v_sample: float | np.ndarray) -> float:
        return self.update(v_sample).frequency_hz
