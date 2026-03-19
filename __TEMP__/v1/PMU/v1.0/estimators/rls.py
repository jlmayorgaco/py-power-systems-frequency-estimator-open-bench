from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np
from .base import BaseEstimator

_EPS = 1e-12


def _clamp(x: float, lo: float, hi: float) -> float:
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    return float(min(max(x, lo), hi))


def _finite_or(x: float, fallback: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else float(fallback)


# --- Helper para suavizado ---
class MovingAverage:
    def __init__(self, win: int):
        self.win = int(max(1, win))
        self.buf = deque(maxlen=self.win)
        self.sum = 0.0

    def reset(self) -> None:
        self.buf.clear()
        self.sum = 0.0

    def step(self, x: float) -> float:
        x = float(x)
        if len(self.buf) == self.win:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        return float(self.sum / max(1, len(self.buf)))


# ============================================================
# Core RLS Engine (AR-2 Model: v[n] + a1*v[n-1] + a2*v[n-2] = 0)
# ============================================================


class AR2RLSEngine:
    """
    Motor RLS adaptativo optimizado para señales de potencia.
    Implementa el modelo de oscilador discreto.
    """

    def __init__(
        self, fs_hz: float, lam: float = 0.99, delta: float = 100.0, f0_hz: float = 60.0
    ):
        self.dt = 1.0 / float(fs_hz)
        self.lam = float(lam)
        self.delta = float(delta)
        self.f0 = float(f0_hz)
        self.reset()

    def reset(self) -> None:
        # Inicialización: a1=-2*cos(w*dt), a2=1 para un oscilador puro
        self.theta = np.array(
            [-2.0 * np.cos(2.0 * np.pi * self.f0 * self.dt), 1.0], dtype=float
        )
        self.P = float(self.delta) * np.eye(2, dtype=float)
        self.phi = np.zeros(2, dtype=float)  # [v[n-1], v[n-2]]
        self._warm = 0  # warm-up de regresor

    def step(self, v_n: float, custom_lam: Optional[float] = None) -> float:
        v_n = float(v_n)

        # Warm-up: llena phi antes de intentar estimar (evita saltos raros al inicio)
        if self._warm < 2:
            self.phi[1] = self.phi[0]
            self.phi[0] = v_n
            self._warm += 1
            return float(self.f0)

        lam = float(custom_lam) if custom_lam is not None else float(self.lam)
        # Guardrail: lam debe estar en (0, 1]
        lam = _clamp(lam, 1e-4, 1.0)

        # 1) Error de predicción
        y_hat = float(-self.phi @ self.theta)
        e = v_n - y_hat

        # 2) Ganancia RLS (proteger denominador)
        P_phi = self.P @ self.phi
        denom = lam + float(self.phi @ P_phi)
        if not np.isfinite(denom) or abs(denom) < _EPS:
            self.reset()
            return float(self.f0)

        g = P_phi / (denom + _EPS)

        # 3) Actualización parámetros y covarianza
        self.theta = self.theta + g * e
        self.P = (self.P - np.outer(g, self.phi) @ self.P) / lam

        # Simetrización numérica
        self.P = 0.5 * (self.P + self.P.T)

        # Guardrail covarianza
        if not np.all(np.isfinite(self.P)):
            self.reset()
            return float(self.f0)

        # 4) Proyección “soft” de estabilidad
        self.theta[1] = float(np.clip(self.theta[1], 0.8, 1.1))

        # Actualizar regresor
        self.phi[1] = self.phi[0]
        self.phi[0] = v_n

        # 5) Extracción de frecuencia:
        a1 = float(self.theta[0])
        a2 = float(self.theta[1])
        a2 = max(a2, 1e-6)

        arg = -a1 / (2.0 * np.sqrt(a2) + _EPS)
        arg = float(np.clip(arg, -1.0, 1.0))

        omega = float(np.arccos(arg))  # rad/sample
        f_hat = omega / (2.0 * np.pi * self.dt)

        if not np.isfinite(f_hat):
            return float(self.f0)

        return float(f_hat)


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
        f0 = float(p.get("f0_hz", 60.0))
        self.engine = AR2RLSEngine(float(p["fs_hz"]), lam=lam, f0_hz=f0)

        self._ma = MovingAverage(int(p.get("win_smooth", 25)))
        self._ma.reset()

    def _step(self, v: float) -> float:
        f_raw = self.engine.step(float(v))
        f_raw = _clamp(_finite_or(f_raw, 60.0), 40.0, 80.0)
        return float(self._ma.step(f_raw))

    @property
    def latency_samples(self) -> int:
        return int(0.5 * int(self._params.get("win_smooth", 25)))


class VFFRLSEstimator(BaseEstimator):
    """
    RLS con Factor de Olvido Variable (VFF).
    Aumenta la velocidad ante transitorios y la precisión en estado estable.
    """

    NAME = "RLS-VFF"
    FAMILY = "Adaptive"

    def reset(self) -> None:
        p = self._params
        f0 = float(p.get("f0_hz", 60.0))
        self.engine = AR2RLSEngine(float(p["fs_hz"]), f0_hz=f0)

        self.lam_min = float(p.get("lam_min", 0.95))
        self.lam_max = float(p.get("lam_max", 0.9999))
        self.lam_min = _clamp(self.lam_min, 1e-4, 1.0)
        self.lam_max = _clamp(self.lam_max, self.lam_min, 1.0)

        self._ma = MovingAverage(int(p.get("win_smooth", 25)))
        self._ma.reset()

        self.e_sq_avg = 0.0

        # Default robusto (evita saturación por escala de voltaje):
        # si el error típico es ~5% p.u., e^2 ~ 0.0025
        self.vff_scale = float(p.get("vff_scale", 0.05 * 0.05))

    def _step(self, v: float) -> float:
        v = float(v)

        # Estimación del error para ajustar lambda
        y_hat = float(-self.engine.phi @ self.engine.theta)
        e = v - y_hat

        # EMA del error cuadrático
        e2 = float(e * e)
        self.e_sq_avg = 0.95 * float(self.e_sq_avg) + 0.05 * e2

        # Lambda adaptativo (blindado)
        x = float(self.e_sq_avg) / max(_EPS, float(self.vff_scale))
        x = _clamp(x, 0.0, 50.0)
        lam = self.lam_max - (self.lam_max - self.lam_min) * float(np.tanh(x))
        lam = _clamp(lam, self.lam_min, self.lam_max)

        f_raw = self.engine.step(v, custom_lam=lam)
        f_raw = _clamp(_finite_or(f_raw, 60.0), 40.0, 80.0)
        return float(self._ma.step(f_raw))

    @property
    def latency_samples(self) -> int:
        return int(0.5 * int(self._params.get("win_smooth", 25)))
