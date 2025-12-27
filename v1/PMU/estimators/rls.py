from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np
from .base import BaseEstimator

# ============================================================
# Core RLS Engine (AR-2 Model: v[n] + a1*v[n-1] + a2*v[n-2] = 0)
# ============================================================

class AR2RLSEngine:
    """
    Motor RLS adaptativo optimizado para señales de potencia.
    Implementa el modelo de oscilador discreto con estabilidad garantizada.
    """
    def __init__(self, fs_hz: float, lam: float = 0.99, delta: float = 100.0):
        self.dt = 1.0 / fs_hz
        self.lam = lam
        self.delta = delta
        self.reset()

    def reset(self) -> None:
        # Inicialización: a1=-2*cos(w*dt), a2=1 para un oscilador puro
        self.theta = np.array([-2.0 * np.cos(2.0 * np.pi * 60.0 * self.dt), 1.0])
        self.P = self.delta * np.eye(2)
        self.phi = np.zeros(2) # [v[n-1], v[n-2]]

    def step(self, v_n: float, custom_lam: Optional[float] = None) -> float:
        lam = custom_lam if custom_lam is not None else self.lam
        
        # 1. Error de predicción (Priori)
        y_hat = float(-self.phi @ self.theta)
        e = v_n - y_hat
        
        # 2. Actualización de ganancia RLS
        P_phi = self.P @ self.phi
        denom = lam + self.phi @ P_phi
        g = P_phi / (denom + 1e-12)
        
        # 3. Actualización de parámetros y covarianza
        self.theta += g * e
        self.P = (self.P - np.outer(g, self.phi) @ self.P) / lam
        
        # Simetrización numérica
        self.P = 0.5 * (self.P + self.P.T)
        
        # 4. Proyección de estabilidad (Polos dentro del círculo unitario)
        # a2 debe estar cerca de 1.0 para oscilaciones sostenidas
        self.theta[1] = np.clip(self.theta[1], 0.8, 1.1)
        
        # Actualizar regresol
        self.phi[1] = self.phi[0]
        self.phi[0] = v_n
        
        # 5. Extracción de frecuencia: f = arccos(-a1 / (2*sqrt(a2))) / (2*pi*dt)
        try:
            arg = -self.theta[0] / (2.0 * np.sqrt(self.theta[1]) + 1e-12)
            omega = np.arccos(np.clip(arg, -1.0, 1.0))
            return omega / (2.0 * np.pi * self.dt)
        except:
            return 60.0

# ============================================================
# Estimadores para Auto-Discovery
# ============================================================

class RLSEstimator(BaseEstimator):
    """
    RLS Estándar con Factor de Olvido Fijo.
    Excelente para señales con SNR constante.
    """
    NAME = "RLS"
    FAMILY = "Adaptive"

    def reset(self) -> None:
        p = self._params
        lam = float(p.get("lambda", 0.999))
        self.engine = AR2RLSEngine(float(p["fs_hz"]), lam=lam)
        self._ma = MovingAverage(int(p.get("win_smooth", 25)))

    def step(self, v: float) -> float:
        f_raw = self.engine.step(v)
        return self._ma.step(np.clip(f_raw, 40.0, 80.0))

    @property
    def latency_samples(self) -> int:
        return int(0.5 * self._params.get("win_smooth", 25))

class VFFRLSEstimator(BaseEstimator):
    """
    RLS con Factor de Olvido Variable (VFF).
    Aumenta la velocidad ante transitorios y la precisión en estado estable.
    """
    NAME = "RLS-VFF"
    FAMILY = "Adaptive"

    def reset(self) -> None:
        p = self._params
        self.engine = AR2RLSEngine(float(p["fs_hz"]))
        self.lam_min = float(p.get("lam_min", 0.95))
        self.lam_max = float(p.get("lam_max", 0.9999))
        self._ma = MovingAverage(int(p.get("win_smooth", 25)))
        self.e_sq_avg = 0.0

    def step(self, v: float) -> float:
        # Estimación del error cuadrático medio para ajustar lambda
        y_hat = -self.engine.phi @ self.engine.theta
        e = v - y_hat
        self.e_sq_avg = 0.95 * self.e_sq_avg + 0.05 * (e**2)
        
        # Lambda adaptativo: ante mucho error (transitorio), lambda baja (olvido rápido)
        lam = self.lam_max - (self.lam_max - self.lam_min) * np.tanh(self.e_sq_avg)
        
        f_raw = self.engine.step(v, custom_lam=lam)
        return self._ma.step(np.clip(f_raw, 40.0, 80.0))

    @property
    def latency_samples(self) -> int:
        return int(0.5 * self._params.get("win_smooth", 25))

# --- Helper para suavizado ---
class MovingAverage:
    def __init__(self, win: int):
        self.win = win
        self.buf = deque(maxlen=win)
        self.sum = 0.0
    def step(self, x: float) -> float:
        if len(self.buf) == self.win: self.sum -= self.buf[0]
        self.buf.append(x); self.sum += x
        return self.sum / len(self.buf)