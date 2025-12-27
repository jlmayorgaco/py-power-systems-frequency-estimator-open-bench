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
    return x % (2.0 * np.pi)

def _huber_weight(u: float, c: float = 1.345) -> float:
    """Función de peso de Huber para mitigar outliers."""
    abs_u = abs(u)
    return 1.0 if abs_u <= c else c / (abs_u + 1e-12)

def _safe_cholesky(P, n):
    """Cholesky robusto con regularización de Tikhonov."""
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        # Regularización si la matriz pierde definición positiva
        P_corr = (P + P.T) / 2 + np.eye(n) * 1e-10
        return np.linalg.cholesky(P_corr)

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
        self.x = np.array([1.0, 0.0])
        self.P = np.eye(2) * 0.1
        self.Q = np.eye(2) * float(self._params.get("q_param", 1e-5))
        self.R = float(self._params.get("r_param", 1e-2))
        self.t_idx = 0
        self.prev_phase = 0.0

    def step(self, z: float) -> float:
        self.P = self.P + self.Q
        phi_nom = self.w0 * self.t_idx * self.dt
        H = np.array([[np.cos(phi_nom), -np.sin(phi_nom)]])
        inn = z - (H @ self.x)[0]
        S = (H @ self.P @ H.T)[0,0] + self.R
        K = (self.P @ H.T) / S
        self.x += K[:,0] * inn
        self.P = (np.eye(2) - K @ H) @ self.P
        phasor_phase = np.arctan2(self.x[1], self.x[0])
        diff = phasor_phase - self.prev_phase
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        self.t_idx += 1
        self.prev_phase = phasor_phase
        return self.f0 + (diff / (2.0 * np.pi * self.dt))

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
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0])
        self.P = np.diag([0.1, 1.0, 0.1])
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4])
        self.R = float(self._params.get("r_param", 1e-2))

    def step(self, z: float) -> float:
        self.x[0] = _wrap_2pi(self.x[0] + self.x[1] * self.dt)
        F = np.eye(3); F[0, 1] = self.dt
        self.P = F @ self.P @ F.T + self.Q
        y_hat = self.x[2] * np.sin(self.x[0])
        H = np.array([[self.x[2]*np.cos(self.x[0]), 0.0, np.sin(self.x[0])]])
        inn = z - y_hat
        S = (H @ self.P @ H.T)[0,0] + self.R
        K = (self.P @ H.T) / S
        self.x += K[:,0] * inn
        self.P = (np.eye(3) - K @ H) @ self.P
        return self.x[1] / (2.0 * np.pi)

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
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0])
        self.P = np.diag([0.1, 1.0, 0.1])
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4])
        self.R = float(self._params.get("r_param", 1e-2))

    def step(self, z: float) -> float:
        self.x[0] = _wrap_2pi(self.x[0] + self.x[1] * self.dt)
        F = np.eye(3); F[0, 1] = self.dt
        P_p = F @ self.P @ F.T + self.Q
        x_p = self.x.copy()
        x_i = x_p.copy()
        for _ in range(3):
            H = np.array([[x_i[2]*np.cos(x_i[0]), 0.0, np.sin(x_i[0])]])
            y_i = x_i[2] * np.sin(x_i[0])
            S = (H @ P_p @ H.T)[0,0] + self.R
            K = (P_p @ H.T) / S
            x_i = x_p + K[:,0] * (z - y_i - (H @ (x_p - x_i))[0])
        self.x = x_i
        H = np.array([[self.x[2]*np.cos(self.x[0]), 0.0, np.sin(self.x[0])]])
        self.P = (np.eye(3) - (P_p @ H.T / ((H @ P_p @ H.T)[0,0] + self.R)) @ H) @ P_p
        return self.x[1] / (2.0 * np.pi)

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
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0])
        self.P = np.eye(3) * 0.1
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-8, q, 1e-4])
        self.R = float(self._params.get("r_param", 1e-2))
        self.alpha, self.kappa, self.beta = 0.1, 0.0, 2.0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        self.Wm = np.full(2*self.n + 1, 1.0/(2*(self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    def _get_sigmas(self, x, P):
        sigmas = np.zeros((2*self.n + 1, self.n))
        S = _safe_cholesky(P, self.n)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + self.gamma * S[:, i]
            sigmas[self.n+i+1] = x - self.gamma * S[:, i]
        return sigmas

    def step(self, z: float) -> float:
        sigmas = self._get_sigmas(self.x, self.P)
        sigmas[:, 0] += sigmas[:, 1] * self.dt
        x_p = np.dot(self.Wm, sigmas)
        x_p[0] = _wrap_2pi(x_p[0])
        P_p = self.Q.copy()
        for i in range(2*self.n+1):
            diff = sigmas[i] - x_p
            P_p += self.Wc[i] * np.outer(diff, diff)
        sigmas_p = self._get_sigmas(x_p, P_p)
        z_sigmas = sigmas_p[:, 2] * np.sin(sigmas_p[:, 0])
        z_p = np.dot(self.Wm, z_sigmas)
        P_zz, P_xz = self.R, np.zeros(self.n)
        for i in range(2*self.n+1):
            dz, dx = z_sigmas[i] - z_p, sigmas_p[i] - x_p
            P_zz += self.Wc[i] * (dz**2)
            P_xz += self.Wc[i] * dx * dz
        K = P_xz / (P_zz + 1e-12)
        self.x = x_p + K * (z - z_p)
        self.P = P_p - np.outer(K, K) * P_zz
        return self.x[1] / (2.0 * np.pi)

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
        self.x = np.array([0.0, 2.0 * np.pi * f0, 1.0])
        self.P = np.eye(3) * 0.01 
        q = float(self._params.get("q_param", 1e-3))
        r = float(self._params.get("r_param", 1e-2))
        self.Q = np.diag([1e-10, q, 1e-6])
        self.R = r
        self.xi = np.sqrt(self.n) * np.concatenate([np.eye(self.n), -np.eye(self.n)], axis=1)

    def step(self, z: float) -> float:
        S = _safe_cholesky(self.P, self.n)
        pts = self.x[:, None] + S @ self.xi
        pts[0, :] = [_wrap_2pi(p) for p in (pts[0, :] + pts[1, :] * self.dt)]
        x_p = np.mean(pts, axis=1)
        err = pts - x_p[:, None]
        P_p = (err @ err.T) / (2 * self.n) + self.Q
        S_p = _safe_cholesky(P_p, self.n)
        pts_p = x_p[:, None] + S_p @ self.xi
        z_pts = pts_p[2, :] * np.sin(pts_p[0, :])
        z_hat = np.mean(z_pts)
        dz = z_pts - z_hat
        P_zz = self.R + np.mean(dz**2)
        P_xz = (pts_p - x_p[:, None]) @ dz / (2 * self.n)
        K = P_xz / (P_zz + 1e-12)
        self.x = x_p + K * (z - z_hat)
        self.P = P_p - np.outer(K, K) * P_zz
        self.x[1] = np.clip(self.x[1], 2*np.pi*40, 2*np.pi*80)
        return self.x[1] / (2.0 * np.pi)

# ============================================================
# 6. EnKF (Ensemble Kalman Filter - Vectorizado)
# ============================================================
class EnKFEstimator(BaseEstimator):
    NAME = "EnKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        self.N = 50
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.X = np.zeros((3, self.N))
        self.X[1, :] = 2.0 * np.pi * f0
        self.X[2, :] = 1.0
        q = float(self._params.get("q_param", 1e-3))
        self.Q_std = np.sqrt([1e-8, q, 1e-4])
        self.R_std = np.sqrt(float(self._params.get("r_param", 1e-2)))

    def step(self, z: float) -> float:
        noise = np.random.normal(0, 1, (3, self.N)) * self.Q_std[:, None]
        self.X[0, :] = _wrap_2pi(self.X[0, :] + self.X[1, :] * self.dt)
        self.X += noise
        z_ens, x_mean = self.X[2, :] * np.sin(self.X[0, :]), np.mean(self.X, axis=1)
        z_mean = np.mean(z_ens)
        E_x, E_z = self.X - x_mean[:, None], z_ens - z_mean
        P_xz = (E_x @ E_z.T) / (self.N - 1)
        P_zz = (E_z @ E_z.T) / (self.N - 1) + self.R_std**2
        K = P_xz / (P_zz + 1e-12)
        z_pert = z + np.random.normal(0, self.R_std, self.N)
        self.X += K[:, None] * (z_pert - z_ens)
        return float(np.mean(self.X[1, :]) / (2.0 * np.pi))

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
        self.x = np.zeros(10)
        self.x[1] = 2.0 * np.pi * f0
        self.x[3] = 1.0 
        self.P = np.eye(10) * 0.1
        
        q = float(self._params.get("q_param", 1e-3))
        # BLOQUE Q REFINADO: Armónicos (x[5]-x[8]) muy penalizados para evitar oscilación
        q_fundamental = q
        q_harmonics = q * 1e-4 # 10,000 veces más pequeño inicialmente
        
        self.Q = np.diag([
            1e-9,           # Fase
            q_fundamental,  # Frecuencia
            q_fundamental*10,# RoCoF
            1e-4, 1e-4,     # Amplitudes Fund (sin/cos)
            q_harmonics, q_harmonics, # Amplitudes 2do
            q_harmonics, q_harmonics, # Amplitudes 3ro
            1e-8            # DC offset
        ])
        self.R = float(self._params.get("r_param", 1e-2))
        self.huber_c = 1.345

    def step(self, z: float) -> float:
        self.x[0] = _wrap_2pi(self.x[0] + self.x[1]*self.dt + 0.5*self.x[2]*self.dt**2)
        self.x[1] += self.x[2]*self.dt
        F = np.eye(10); F[0,1]=self.dt; F[0,2]=0.5*self.dt**2; F[1,2]=self.dt
        self.P = F @ self.P @ F.T + self.Q
        th = self.x[0]
        s1, c1, s2, c2, s3, c3 = np.sin(th), np.cos(th), np.sin(2*th), np.cos(2*th), np.sin(3*th), np.cos(3*th)
        y_hat = (self.x[3]*s1 + self.x[4]*c1 + self.x[5]*s2 + self.x[6]*c2 + self.x[7]*s3 + self.x[8]*c3 + self.x[9])
        H = np.zeros((1, 10))
        H[0, 0] = self.x[3]*c1 - self.x[4]*s1 + 2*(self.x[5]*c2 - self.x[6]*s2) + 3*(self.x[7]*c3 - self.x[8]*s3)
        H[0, 3:] = [s1, c1, s2, c2, s3, c3, 1.0]
        inn = z - y_hat
        S = (H @ self.P @ H.T)[0,0] + self.R
        
        # Robustificación Huber
        w_h = _huber_weight(inn / np.sqrt(S + 1e-12), self.huber_c)
        R_eff = self.R / (w_h**2)
        
        K = (self.P @ H.T) / ((H @ self.P @ H.T)[0,0] + R_eff)
        self.x += K[:,0] * inn
        self.P = (np.eye(10) - K @ H) @ self.P
        # Adaptación de R
        self.R = 0.98 * self.R + 0.02 * (inn**2)
        return self.x[1] / (2.0 * np.pi)

# ============================================================
# 8. RA-EKF (Tu propuesta: Robusto y Adaptativo 4 estados)
# ============================================================
class RAEKFEstimator(BaseEstimator):
    NAME = "RA-EKF"
    FAMILY = "Kalman"

    def reset(self) -> None:
        fs = float(self._params["fs_hz"])
        self.dt = 1.0 / fs
        f0 = float(self._params.get("f0_hz", 60.0))
        self.x = np.array([0.0, 2.0 * np.pi * f0, 0.0, 1.0])
        self.P = np.diag([0.1, 1.0, 10.0, 0.1])
        q = float(self._params.get("q_param", 1e-3))
        self.Q = np.diag([1e-9, q, q*10, 1e-4])
        self.R = float(self._params.get("r_param", 1e-2))
        self.huber_c = 1.345

    def step(self, z: float) -> float:
        self.x[0] = _wrap_2pi(self.x[0] + self.x[1]*self.dt + 0.5*self.x[2]*self.dt**2)
        self.x[1] += self.x[2]*self.dt
        F = np.eye(4); F[0,1]=self.dt; F[0,2]=0.5*self.dt**2; F[1,2]=self.dt
        self.P = F @ self.P @ F.T + self.Q
        th = self.x[0]
        y_hat = self.x[3] * np.sin(th)
        H = np.array([[self.x[3]*np.cos(th), 0.0, 0.0, np.sin(th)]])
        inn = z - y_hat
        S = (H @ self.P @ H.T)[0,0] + self.R
        w_h = _huber_weight(inn / np.sqrt(S + 1e-12), self.huber_c)
        R_eff = self.R / (w_h**2)
        K = (self.P @ H.T) / ((H @ self.P @ H.T)[0,0] + R_eff)
        self.x += K[:,0] * inn
        self.P = (np.eye(4) - K @ H) @ self.P
        self.R = 0.99 * self.R + 0.01 * (inn**2)
        return self.x[1] / (2.0 * np.pi)