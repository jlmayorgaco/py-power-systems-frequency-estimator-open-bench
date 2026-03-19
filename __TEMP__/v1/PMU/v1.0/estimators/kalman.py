from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

from .base import BaseEstimator

# ============================================================
# Matemáticas Robustas y Utilidades
# ============================================================


def _wrap_2pi(x):
    """
    Mantiene la fase en [0, 2pi).
    Vectorizado para soportar EnKF y escalares.
    """
    return np.asarray(x) % (2.0 * np.pi)


def _angle_diff(a, b):
    """
    Diferencia angular envuelta a (-pi, pi]. Vectorizado.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _huber_weight(u: float, c: float = 1.345) -> float:
    """Función de peso de Huber para mitigar outliers."""
    abs_u = abs(float(u))
    return 1.0 if abs_u <= c else c / (abs_u + 1e-12)


def _safe_cholesky(P, n):
    """Cholesky robusto con regularización de Tikhonov."""
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        # Regularización si la matriz pierde definición positiva
        P_sym = (P + P.T) / 2.0
        # Escala un poco más robusta (evita fallas si P tiene autovalores negativos pequeños)
        jitter = 1e-10 * max(1.0, float(np.trace(P_sym)) / max(1, n))
        P_corr = P_sym + np.eye(n) * jitter
        return np.linalg.cholesky(P_corr)


def _circular_mean(angles, weights=None):
    """
    Media circular robusta (evita promediar fase linealmente).
    angles en rad.
    """
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        s = np.mean(np.sin(angles))
        c = np.mean(np.cos(angles))
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.sum(w) + 1e-18)
        s = np.sum(w * np.sin(angles))
        c = np.sum(w * np.cos(angles))
    return float(np.arctan2(s, c))


# ============================================================
# 1. LKF (Linear Kalman Filter)
# ============================================================
class LKFEstimator(BaseEstimator):
    NAME = "LKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        self.f0 = float(self._params.get("f0_hz", 60.0))
        self.w0 = 2.0 * np.pi * self.f0
        self.x = np.array([1.0, 0.0], dtype=float)
        self.P = np.eye(2, dtype=float) * 0.1
        self.Q = np.eye(2, dtype=float) * float(self._params.get("q_param", 1e-5))
        self.R = float(self._params.get("r_param", 1e-2))
        self.t_idx = 0
        self.prev_phase = 0.0

    def _step(self, z: float) -> float:
        # Predict (random-walk phasor)
        self.P = self.P + self.Q

        # Measurement model: z = [cos(phi_nom), -sin(phi_nom)] x + noise
        phi_nom = self.w0 * self.t_idx * self.dt
        H = np.array([[np.cos(phi_nom), -np.sin(phi_nom)]], dtype=float)

        inn = float(z) - float((H @ self.x)[0])
        S = float((H @ self.P @ H.T)[0, 0] + self.R)
        S = max(S, 1e-18)

        K = (self.P @ H.T) / S  # (2x1)
        self.x = self.x + K[:, 0] * inn
        self.P = (np.eye(2) - K @ H) @ self.P

        # Frequency from wrapped phase increment
        phasor_phase = float(np.arctan2(self.x[1], self.x[0]))
        diff = float(_angle_diff(phasor_phase, self.prev_phase))

        self.t_idx += 1
        self.prev_phase = phasor_phase

        f_hat = self.f0 + (diff / (2.0 * np.pi * self.dt))
        return float(f_hat)


# ============================================================
# 2. EKF (Extended Kalman Filter - 3 States)
# ============================================================
class EKFEstimator(BaseEstimator):
    NAME = "EKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.x = np.array(
            [0.0, 2.0 * np.pi * f0, 1.0], dtype=float
        )  # [theta, omega, A]
        self.P = np.diag([0.1, 1.0, 0.1]).astype(float)
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4]).astype(float)
        self.R = float(self._params.get("r_param", 1e-2))

    def _step(self, z: float) -> float:
        # Predict
        self.x[0] = float(_wrap_2pi(self.x[0] + self.x[1] * self.dt))
        F = np.eye(3, dtype=float)
        F[0, 1] = self.dt
        self.P = F @ self.P @ F.T + self.Q

        # Update
        y_hat = float(self.x[2] * np.sin(self.x[0]))
        H = np.array(
            [[self.x[2] * np.cos(self.x[0]), 0.0, np.sin(self.x[0])]], dtype=float
        )

        inn = float(z) - y_hat
        S = float((H @ self.P @ H.T)[0, 0] + self.R)
        S = max(S, 1e-18)
        K = (self.P @ H.T) / S

        self.x = self.x + K[:, 0] * inn
        self.P = (np.eye(3) - K @ H) @ self.P

        # Guardrails (opcional, evita blow-up silencioso)
        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        self.x[2] = float(np.clip(self.x[2], 1e-6, 10.0))

        return float(self.x[1] / (2.0 * np.pi))


# ============================================================
# 3. IEKF (Iterated Extended Kalman Filter)
# ============================================================
class IEKFEstimator(BaseEstimator):
    NAME = "IEKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0], dtype=float)
        self.P = np.diag([0.1, 1.0, 0.1]).astype(float)
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4]).astype(float)
        self.R = float(self._params.get("r_param", 1e-2))

    def _step(self, z: float) -> float:
        # Predict
        self.x[0] = float(_wrap_2pi(self.x[0] + self.x[1] * self.dt))
        F = np.eye(3, dtype=float)
        F[0, 1] = self.dt
        P_p = F @ self.P @ F.T + self.Q
        x_p = self.x.copy()

        # Iterate measurement update
        x_i = x_p.copy()
        for _ in range(3):
            H = np.array([[x_i[2] * np.cos(x_i[0]), 0.0, np.sin(x_i[0])]], dtype=float)
            y_i = float(x_i[2] * np.sin(x_i[0]))
            S = float((H @ P_p @ H.T)[0, 0] + self.R)
            S = max(S, 1e-18)
            K = (P_p @ H.T) / S  # 3x1

            # IEKF correction (keep consistent linearization)
            x_i = x_p + K[:, 0] * (float(z) - y_i - float((H @ (x_p - x_i))[0]))

        self.x = x_i
        H = np.array(
            [[self.x[2] * np.cos(self.x[0]), 0.0, np.sin(self.x[0])]], dtype=float
        )
        S = float((H @ P_p @ H.T)[0, 0] + self.R)
        S = max(S, 1e-18)
        K = (P_p @ H.T) / S
        self.P = (np.eye(3) - K @ H) @ P_p

        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        self.x[2] = float(np.clip(self.x[2], 1e-6, 10.0))

        return float(self.x[1] / (2.0 * np.pi))


# ============================================================
# 4. UKF (Unscented Kalman Filter - Blindado)
# ============================================================
class UKFEstimator(BaseEstimator):
    NAME = "UKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        self.n = 3
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0], dtype=float)
        self.P = np.eye(3, dtype=float) * 0.1
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4]).astype(float)
        self.R = float(self._params.get("r_param", 1e-2))

        # UT params
        self.alpha = float(self._params.get("ukf_alpha", 0.1))
        self.kappa = float(self._params.get("ukf_kappa", 0.0))
        self.beta = float(self._params.get("ukf_beta", 2.0))
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = float(np.sqrt(self.n + self.lam))

        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)), dtype=float)
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - self.alpha**2 + self.beta)

    def _get_sigmas(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n), dtype=float)
        S = _safe_cholesky(P, self.n)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i + 1] = x + self.gamma * S[:, i]
            sigmas[self.n + i + 1] = x - self.gamma * S[:, i]
        return sigmas

    def _step(self, z: float) -> float:
        # Sigma points
        sigmas = self._get_sigmas(self.x, self.P)

        # Propagate through process model
        sigmas[:, 0] = _wrap_2pi(sigmas[:, 0] + sigmas[:, 1] * self.dt)

        # Mean: circular for theta, linear for others
        theta_mean = _circular_mean(sigmas[:, 0], self.Wm)
        omega_mean = float(np.dot(self.Wm, sigmas[:, 1]))
        A_mean = float(np.dot(self.Wm, sigmas[:, 2]))
        x_p = np.array([theta_mean, omega_mean, A_mean], dtype=float)

        # Covariance
        P_p = self.Q.copy()
        for i in range(2 * self.n + 1):
            d = sigmas[i] - x_p
            d[0] = float(_angle_diff(sigmas[i, 0], x_p[0]))
            P_p += self.Wc[i] * np.outer(d, d)

        # Measurement prediction
        sigmas_p = self._get_sigmas(x_p, P_p)
        z_sigmas = sigmas_p[:, 2] * np.sin(sigmas_p[:, 0])
        z_p = float(np.dot(self.Wm, z_sigmas))

        P_zz = float(self.R)
        P_xz = np.zeros(self.n, dtype=float)
        for i in range(2 * self.n + 1):
            dz = float(z_sigmas[i] - z_p)
            dx = sigmas_p[i] - x_p
            dx[0] = float(_angle_diff(sigmas_p[i, 0], x_p[0]))
            P_zz += self.Wc[i] * (dz**2)
            P_xz += self.Wc[i] * dx * dz

        P_zz = max(P_zz, 1e-18)
        K = P_xz / P_zz

        # Update
        self.x = x_p + K * (float(z) - z_p)
        self.x[0] = float(_wrap_2pi(self.x[0]))
        self.P = P_p - np.outer(K, K) * P_zz

        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        self.x[2] = float(np.clip(self.x[2], 1e-6, 10.0))

        return float(self.x[1] / (2.0 * np.pi))


# ============================================================
# 5. CKF (Cubature Kalman Filter - Blindado)
# ============================================================
class CKFEstimator(BaseEstimator):
    NAME = "CKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        self.n = 3
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0], dtype=float)
        self.P = np.eye(3, dtype=float) * 0.01
        q = float(self._params.get("q_param", 1e-3))
        r = float(self._params.get("r_param", 1e-2))
        self.Q = np.diag([1e-10, q, 1e-6]).astype(float)
        self.R = float(r)
        self.xi = np.sqrt(self.n) * np.concatenate(
            [np.eye(self.n), -np.eye(self.n)], axis=1
        )  # (n, 2n)

    def _step(self, z: float) -> float:
        # Cubature points
        S = _safe_cholesky(self.P, self.n)
        pts = self.x[:, None] + S @ self.xi  # (n, 2n)

        # Propagate: wrap theta after propagation
        pts[0, :] = _wrap_2pi(pts[0, :] + pts[1, :] * self.dt)

        # Mean: circular for theta, linear for others
        theta_mean = _circular_mean(pts[0, :])
        omega_mean = float(np.mean(pts[1, :]))
        A_mean = float(np.mean(pts[2, :]))
        x_p = np.array([theta_mean, omega_mean, A_mean], dtype=float)

        # Covariance
        err = pts - x_p[:, None]
        err[0, :] = _angle_diff(pts[0, :], x_p[0])
        P_p = (err @ err.T) / (2 * self.n) + self.Q

        # Predicted measurement
        S_p = _safe_cholesky(P_p, self.n)
        pts_p = x_p[:, None] + S_p @ self.xi
        z_pts = pts_p[2, :] * np.sin(pts_p[0, :])
        z_hat = float(np.mean(z_pts))

        dz = z_pts - z_hat
        P_zz = float(self.R + np.mean(dz**2))

        dx = pts_p - x_p[:, None]
        dx[0, :] = _angle_diff(pts_p[0, :], x_p[0])
        P_xz = (dx @ dz) / (2 * self.n)  # (n,)

        P_zz = max(P_zz, 1e-18)
        K = P_xz / P_zz

        # Update
        self.x = x_p + K * (float(z) - z_hat)
        self.x[0] = float(_wrap_2pi(self.x[0]))
        self.P = P_p - np.outer(K, K) * P_zz

        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        self.x[2] = float(np.clip(self.x[2], 1e-6, 10.0))

        return float(self.x[1] / (2.0 * np.pi))


# ============================================================
# 6. EnKF (Ensemble Kalman Filter - Vectorizado)
# ============================================================
class EnKFEstimator(BaseEstimator):
    NAME = "EnKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        self.N = int(self._params.get("enkf_N", 50))
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))

        self.X = np.zeros((3, self.N), dtype=float)
        self.X[0, :] = 0.0
        self.X[1, :] = 2.0 * np.pi * f0
        self.X[2, :] = 1.0

        q = float(self._params.get("q_param", 1e-3))
        self.Q_std = np.sqrt(np.array([1e-8, q, 1e-4], dtype=float))
        self.R_std = float(np.sqrt(float(self._params.get("r_param", 1e-2))))

        # Reproducibilidad: RNG local (no toca np.random global)
        seed = self._params.get("seed", None)
        if seed is None:
            # fallback razonable si no te pasan seed: fija para determinismo por defecto
            seed = 0
        # Mezcla con NAME para evitar que distintos métodos compartan misma secuencia si no quieres
        mix = abs(hash(self.NAME)) % 1_000_000
        self.rng = np.random.default_rng(int(seed) + int(mix))

    def _step(self, z: float) -> float:
        # Forecast
        noise = self.rng.normal(0.0, 1.0, (3, self.N)) * self.Q_std[:, None]
        self.X[0, :] = _wrap_2pi(self.X[0, :] + self.X[1, :] * self.dt)
        self.X += noise

        # Predicted measurements
        z_ens = self.X[2, :] * np.sin(self.X[0, :])
        x_mean = np.mean(self.X, axis=1)
        z_mean = float(np.mean(z_ens))

        # Anomalies
        E_x = self.X - x_mean[:, None]
        E_x[0, :] = _angle_diff(
            self.X[0, :], _circular_mean(self.X[0, :])
        )  # theta anomalies
        E_z = z_ens - z_mean

        # Cross-covariances (scalars here)
        P_xz = (E_x @ E_z.T) / max(1, (self.N - 1))  # (3,)
        P_zz = float((E_z @ E_z.T) / max(1, (self.N - 1)) + self.R_std**2)

        P_zz = max(P_zz, 1e-18)
        K = P_xz / P_zz  # (3,)

        # Stochastic EnKF update
        z_pert = float(z) + self.rng.normal(0.0, self.R_std, self.N)
        self.X += K[:, None] * (z_pert - z_ens)

        # Wrap theta after update
        self.X[0, :] = _wrap_2pi(self.X[0, :])

        # Output: mean omega
        omega_hat = float(np.mean(self.X[1, :]))
        omega_hat = float(np.clip(omega_hat, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        return float(omega_hat / (2.0 * np.pi))


# ============================================================
# 7. RA-EKF2 (Robusto-Adaptativo 10 estados)
# ============================================================
class RAEKF2Estimator(BaseEstimator):
    NAME = "RA-EKF2"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))

        self.x = np.zeros(10, dtype=float)
        self.x[1] = 2.0 * np.pi * f0

        # [CORRECCIÓN CRÍTICA]
        # x[3] es coef Sin, x[4] es coef Cos.
        # Las redes suelen ser tipo Cos. Si inicializamos x[3]=1 (Sin),
        # el modelo empieza con error de fase de 90° y y_hat=0 vs y=1.
        # Esto dispara R a infinito y mata el filtro.
        self.x[3] = 0.0  # Sin coeff
        self.x[4] = 1.0  # Cos coeff -> Asume start en peak (común en sim)

        # Inicializamos P con alta incertidumbre en armónicos para que
        # se adapte rápido si la fase inicial no es exacta.
        self.P = np.eye(10, dtype=float) * 0.1
        self.P[3:, 3:] = np.eye(7) * 10.0  # Alta varianza en coeficientes

        q = float(self._params.get("q_param", 1e-3))
        q_fund = q
        q_harm = q * 1e-4

        self.Q = np.diag(
            [
                1e-9,  # theta
                q_fund,  # omega
                q_fund * 10.0,  # rocof
                1e-3,
                1e-3,  # fund sin/cos (flexible para amplitud)
                q_harm,
                q_harm,  # 2nd
                q_harm,
                q_harm,  # 3rd
                1e-8,  # DC
            ]
        ).astype(float)

        self.R = float(self._params.get("r_param", 1e-2))
        self.huber_c = float(self._params.get("huber_c", 1.345))
        self.R_min = float(self._params.get("r_min", 1e-10))
        self.R_max = float(self._params.get("r_max", 1e2))

    def _step(self, z: float) -> float:
        # Predict
        self.x[0] = float(
            _wrap_2pi(self.x[0] + self.x[1] * self.dt + 0.5 * self.x[2] * self.dt**2)
        )
        self.x[1] = float(self.x[1] + self.x[2] * self.dt)

        F = np.eye(10, dtype=float)
        F[0, 1] = self.dt
        F[0, 2] = 0.5 * self.dt**2
        F[1, 2] = self.dt
        self.P = F @ self.P @ F.T + self.Q

        # Measurement
        th = self.x[0]
        s1, c1 = np.sin(th), np.cos(th)
        s2, c2 = np.sin(2.0 * th), np.cos(2.0 * th)
        s3, c3 = np.sin(3.0 * th), np.cos(3.0 * th)

        y_hat = (
            self.x[3] * s1
            + self.x[4] * c1
            + self.x[5] * s2
            + self.x[6] * c2
            + self.x[7] * s3
            + self.x[8] * c3
            + self.x[9]
        )

        H = np.zeros((1, 10), dtype=float)
        # Chain rule d/dTheta: (d/dt) of (A*sin(wt + phi))
        # d(x3*s1 + x4*c1)/dth = x3*c1 - x4*s1
        H[0, 0] = (
            self.x[3] * c1
            - self.x[4] * s1
            + 2.0 * (self.x[5] * c2 - self.x[6] * s2)
            + 3.0 * (self.x[7] * c3 - self.x[8] * s3)
        )
        H[0, 3:] = [s1, c1, s2, c2, s3, c3, 1.0]

        inn = float(z) - float(y_hat)
        S = float((H @ self.P @ H.T)[0, 0] + self.R)
        S = max(S, 1e-18)

        # Huber
        w_h = _huber_weight(inn / np.sqrt(S + 1e-12), self.huber_c)
        R_eff = float(self.R) / (w_h**2)

        denom = float((H @ self.P @ H.T)[0, 0] + R_eff)
        denom = max(denom, 1e-18)
        K = (self.P @ H.T) / denom

        self.x = self.x + K[:, 0] * inn
        self.P = (np.eye(10) - K @ H) @ self.P

        # [CORRECCIÓN]: Adaptación más suave para evitar explosión inicial
        # Antes: 0.98/0.02 -> Ahora: 0.995/0.005
        # También clippeamos 'inn' para que un error gigante inicial no rompa R
        inn_sq_clamped = min(inn**2, 100.0 * self.R)
        self.R = float(
            np.clip(0.995 * self.R + 0.005 * inn_sq_clamped, self.R_min, self.R_max)
        )

        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        return float(self.x[1] / (2.0 * np.pi))


# ============================================================
# 8. RA-EKF
# ============================================================
class RAEKFEstimator(BaseEstimator):
    NAME = "RA-EKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))

        self.x = np.array([0.0, 2.0 * np.pi * f0, 0.0, 1.0], dtype=float)
        self.P = np.diag([0.1, 1.0, 10.0, 0.1]).astype(float)

        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-9, q, q * 10.0, 1e-4]).astype(float)

        self.R = float(self._params.get("r_param", 1e-2))
        self.huber_c = float(self._params.get("huber_c", 1.345))
        self.R_min = float(self._params.get("r_min", 1e-10))
        self.R_max = float(self._params.get("r_max", 1e2))

    def _step(self, z: float) -> float:
        self.x[0] = float(
            _wrap_2pi(self.x[0] + self.x[1] * self.dt + 0.5 * self.x[2] * self.dt**2)
        )
        self.x[1] = float(self.x[1] + self.x[2] * self.dt)

        F = np.eye(4, dtype=float)
        F[0, 1] = self.dt
        F[0, 2] = 0.5 * self.dt**2
        F[1, 2] = self.dt
        self.P = F @ self.P @ F.T + self.Q

        th = self.x[0]
        y_hat = float(self.x[3] * np.sin(th))
        H = np.array([[self.x[3] * np.cos(th), 0.0, 0.0, np.sin(th)]], dtype=float)

        inn = float(z) - y_hat
        S = float((H @ self.P @ H.T)[0, 0] + self.R)
        S = max(S, 1e-18)

        w_h = _huber_weight(inn / np.sqrt(S + 1e-12), self.huber_c)
        R_eff = float(self.R) / (w_h**2)

        denom = float((H @ self.P @ H.T)[0, 0] + R_eff)
        denom = max(denom, 1e-18)
        K = (self.P @ H.T) / denom

        self.x = self.x + K[:, 0] * inn
        self.P = (np.eye(4) - K @ H) @ self.P

        self.R = float(np.clip(0.99 * self.R + 0.01 * (inn**2), self.R_min, self.R_max))

        self.x[1] = float(np.clip(self.x[1], 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))
        self.x[3] = float(np.clip(self.x[3], 1e-6, 10.0))
        return float(self.x[1] / (2.0 * np.pi))
