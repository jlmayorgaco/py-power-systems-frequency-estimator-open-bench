"""
estimators/monophasic/f1_kalman/ekf_freq.py  [CANONICAL]

EKFFreqEstimator — Extended Kalman Filter for single-phase frequency tracking.

State model
───────────
  State vector:  x = [φ, ω]   (phase rad, angular freq rad/s)
  Process model: φ[n] = φ[n-1] + ω[n-1]·Ts
                 ω[n] = ω[n-1]             (constant-frequency model)
  Measurement:   z[n] = A·sin(φ[n]) + noise   where A is amplitude estimate

Linearisation
─────────────
  Jacobian of measurement model:
    H = [∂z/∂φ, ∂z/∂ω] = [A·cos(φ), 0]

  A is tracked adaptively as an exponential moving average of |v|·√2.

Numerical stability
───────────────────
  · Joseph form P update  →  P stays symmetric positive-semi-definite.
  · P entries clamped to ±1e8  →  prevents covariance explosion on weak signals.
  · ω clamped to [0.5·MIN, 2·MAX] Hz  →  prevents unbounded drift on absent signal.
  · S = H P Hᵀ + R is always finite and positive.

References
──────────
  Brown, R.G. & Hwang, P.Y.C. (1997). Introduction to Random Signals
  and Applied Kalman Filtering, 3rd ed.  Wiley.

  Dash, P.K., Pradhan, A.K., Panda, G. (1999). "Frequency estimation of
  distorted power system signals using extended complex Kalman filter."
  IEEE Trans. Power Del., 14(3), 761-766.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class EKFFreqEstimator(BaseEstimator):
    """
    EKF frequency estimator — 2-state (phase + angular frequency).

    Optimal for Gaussian noise; handles smooth frequency variations well.
    Less suited to abrupt frequency steps without re-initialisation.
    """

    SPEC = EstimatorSpec(
        name="EKF_Freq",
        family="Kalman",
        family_path="monophasic/f1_kalman",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"fs": 10_000.0, "q_omega": 1.0, "r": 0.01}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="q_omega",
                    default=1.0,
                    type="float",
                    values=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                    description=(
                        "Process noise std-dev on angular frequency (rad/s / sample). "
                        "Larger → faster adaptation to frequency changes but noisier output."
                    ),
                ),
                TuningParam(
                    name="r",
                    default=0.01,
                    type="float",
                    values=[0.0001, 0.001, 0.01, 0.1, 1.0],
                    description=(
                        "Measurement noise variance (V²).  Should reflect the "
                        "expected noise power on the voltage signal."
                    ),
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self._fs = fs
        self._Ts = 1.0 / fs
        self._q_w = float(self._config.get("q_omega", 1.0))
        self._r = float(self._config.get("r", 0.01))

        # State: [phi, omega]
        self._x = np.array([0.0, 2.0 * math.pi * self.NOMINAL_FREQ_HZ])

        # Covariance (2x2); Joseph-form update keeps it PSD
        self._P = np.eye(2) * 10.0

        # Amplitude estimate (exponential moving average of |v|·√2)
        self._A = 1.0
        self._alpha_amp = 0.01  # smoothing coefficient
        self._f_est = self.NOMINAL_FREQ_HZ
        self._n_samples: int = 0

    def structural_latency_samples(self) -> int:
        fs = float(self._config.get("fs", 10_000.0))
        return int(2 * fs / self.NOMINAL_FREQ_HZ)  # ≈ 2 fundamental cycles

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        self._n_samples += 1
        try:
            f = self._ekf_step(float(voltage))
        except Exception:
            return EstimatorOutput(frequency_hz=float("nan"), valid=False)

        valid = (
            math.isfinite(f)
            and self._n_samples > self.structural_latency_samples()
            and self.MIN_VALID_FREQ_HZ <= f <= self.MAX_VALID_FREQ_HZ
        )
        return EstimatorOutput(
            frequency_hz=f,
            valid=valid,
            phase_rad=float(self._x[0]),
            amplitude_pu=self._A,
        )

    # ── Internal EKF algorithm ─────────────────────────────────────────────────

    def _ekf_step(self, v: float) -> float:
        Ts = self._Ts
        q_w = self._q_w
        r = self._r
        A = self._A

        phi, omega = self._x

        # ── Predict ───────────────────────────────────────────────────────────
        phi_pred = phi + omega * Ts
        omega_pred = omega

        F = np.array([[1.0, Ts], [0.0, 1.0]])
        Q = np.array([[0.0, 0.0], [0.0, q_w**2]])

        P_pred = F @ self._P @ F.T + Q
        np.clip(P_pred, -1e8, 1e8, out=P_pred)  # prevent covariance blow-up

        # ── Update ────────────────────────────────────────────────────────────
        z_pred = A * math.sin(phi_pred)
        innov = v - z_pred

        H = np.array([[A * math.cos(phi_pred), 0.0]])

        S = float((H @ P_pred @ H.T).item()) + r
        K = (P_pred @ H.T) / S

        self._x = np.array([phi_pred, omega_pred]) + K.ravel() * innov

        IKH = np.eye(2) - K @ H
        self._P = IKH @ P_pred @ IKH.T + r * (K @ K.T)  # Joseph form

        # Wrap phase to [0, 2π)
        self._x[0] = self._x[0] % (2.0 * math.pi)

        # Keep angular frequency in physical range
        _omega_lo = 2.0 * math.pi * (self.MIN_VALID_FREQ_HZ * 0.5)
        _omega_hi = 2.0 * math.pi * (self.MAX_VALID_FREQ_HZ * 2.0)
        self._x[1] = max(_omega_lo, min(_omega_hi, self._x[1]))

        # Update amplitude estimate
        self._A += self._alpha_amp * (abs(v) * math.sqrt(2.0) - self._A)
        self._A = max(1e-3, self._A)

        self._f_est = self._x[1] / (2.0 * math.pi)
        return self._f_est

    # Backward-compat
    def _step(self, v_sample: float) -> float:
        return self._ekf_step(float(v_sample))
