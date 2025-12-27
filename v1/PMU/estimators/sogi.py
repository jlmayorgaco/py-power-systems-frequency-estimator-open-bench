from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple
import numpy as np
from .base import BaseEstimator

# ============================================================
# Core Utilities
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

class MovingAverage:
    """Filtro de media móvil eficiente para suavizado de frecuencia."""
    def __init__(self, win: int):
        self.win = int(max(1, win))
        self.buf: Deque[float] = deque(maxlen=self.win)
        self.sum = 0.0

    def reset(self) -> None:
        self.buf.clear()
        self.sum = 0.0

    def step(self, x: float) -> float:
        if len(self.buf) == self.win:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        return self.sum / len(self.buf)

# ============================================================
# 1) SOGI-QSG (Tustin-based)
# ============================================================

class SOGI_QSG:
    """
    Quadrature Signal Generator basado en SOGI.
    Implementa integración trapezoidal para máxima precisión espectral.
    """
    def __init__(self, fs_hz: float, k: float = 1.414):
        self.dt = 1.0 / fs_hz
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.va_dot_old = 0.0
        self.vb_dot_old = 0.0

    def step(self, v_in: float, w: float) -> Tuple[float, float, float]:
        # Error de entrada
        e = v_in - self.v_alpha
        
        # Cálculo de derivadas actuales
        va_dot = self.k * w * e - w * self.v_beta
        vb_dot = w * self.v_alpha
        
        # Integración Trapezoidal (Tustin)
        self.v_alpha += (self.dt / 2.0) * (va_dot + self.va_dot_old)
        self.v_beta += (self.dt / 2.0) * (vb_dot + self.vb_dot_old)
        
        # Almacenar derivadas para el siguiente paso
        self.va_dot_old = va_dot
        self.vb_dot_old = vb_dot
        
        return self.v_alpha, self.v_beta, e

# ============================================================
# 2) Classic SOGI-FLL (Standard & Industrial)
# ============================================================

class ClassicSOGIFLL:
    """Lazo de frecuencia basado en SOGI con normalización opcional."""
    def __init__(self, fs_hz: float, f0_hz: float, gamma: float, k: float, use_amp_norm: bool = True):
        self.fs = fs_hz
        self.f0 = f0_hz
        self.dt = 1.0 / fs_hz
        self.gamma = gamma
        self.use_amp_norm = use_amp_norm
        
        self.qsg = SOGI_QSG(fs_hz, k)
        self.w = 2.0 * np.pi * f0_hz
        self.A2 = 1.0  # Estimador de amplitud cuadrado
        self.alpha_amp = (self.dt) / (0.02 + self.dt) # LPF de 20ms para amplitud

    def reset(self) -> None:
        self.qsg.reset()
        self.w = 2.0 * np.pi * self.f0
        self.A2 = 1.0

    def step(self, v_in: float) -> float:
        va, vb, e = self.qsg.step(v_in, self.w)
        
        # Estimación de Amplitud (LPF sobre va^2 + vb^2)
        inst_a2 = va**2 + vb**2
        self.A2 = (1.0 - self.alpha_amp) * self.A2 + self.alpha_amp * inst_a2
        
        # Ley de adaptación FLL
        denom = (self.A2 + 1e-6) if self.use_amp_norm else 1.0
        dw = -self.gamma * e * vb / denom
        
        # Anti-windup y límites de frecuencia
        self.w = _clamp(self.w + self.dt * dw, 2*np.pi*40, 2*np.pi*80)
        return self.w / (2.0 * np.pi)

# ============================================================
# 3) MSOGI-FLL (Parallel Harmonic Rejection)
# ============================================================

class MSOGIFLL:
    """SOTA: Multi-Resonant SOGI para rechazo de armónicos en microredes."""
    def __init__(self, fs_hz: float, f0_hz: float, gamma: float, harmonics: Sequence[int]):
        self.fund = ClassicSOGIFLL(fs_hz, f0_hz, gamma, 1.414, True)
        self.h_orders = harmonics
        self.h_qsgs = [SOGI_QSG(fs_hz, k=1.0) for _ in harmonics]

    def reset(self) -> None:
        self.fund.reset()
        for qsg in self.h_qsgs: qsg.reset()

    def step(self, v_in: float) -> float:
        w_est = self.fund.w
        
        # Cancelación en paralelo: extraemos armónicos de la señal bruta
        v_harm = 0.0
        for h, qsg in zip(self.h_orders, self.h_qsgs):
            va_h, _, _ = qsg.step(v_in, h * w_est)
            v_harm += va_h
            
        # El lazo principal ve la señal "limpia"
        return self.fund.step(v_in - v_harm)

# ============================================================
# 4) Estimator Wrappers
# ============================================================

class SOGIClassicEstimator(BaseEstimator):
    NAME = "SOGI-Classic"
    FAMILY = "SOGI-FLL"
    def reset(self):
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = ClassicSOGIFLL(p["fs_hz"], p.get("f0_hz", 60), p.get("g", 100), p.get("k", 1.414), False)
    def step(self, v): return self._ma.step(self._impl.step(v))
    @property
    def latency_samples(self): return int(0.8 * (self._params["fs_hz"] / 60))

class SOGIIndustrialEstimator(BaseEstimator):
    NAME = "SOGI-Industrial"
    FAMILY = "SOGI-FLL"
    def reset(self):
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = ClassicSOGIFLL(p["fs_hz"], p.get("f0_hz", 60), p.get("g", 100), p.get("k", 1.414), True)
    def step(self, v): return self._ma.step(self._impl.step(v))
    @property
    def latency_samples(self): return int(0.9 * (self._params["fs_hz"] / 60))

class MSOGIFLLEstimator(BaseEstimator):
    NAME = "MSOGI-FLL"
    FAMILY = "SOGI-FLL"
    def reset(self):
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = MSOGIFLL(p["fs_hz"], p.get("f0_hz", 60), p.get("g", 100), p.get("harmonics", (3,5,7)))
    def step(self, v): return self._ma.step(self._impl.step(v))
    @property
    def latency_samples(self): return int(1.1 * (self._params["fs_hz"] / 60))