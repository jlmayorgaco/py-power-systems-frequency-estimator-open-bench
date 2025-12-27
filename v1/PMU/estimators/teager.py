from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np
from .base import BaseEstimator

# ============================================================
# Robust Signal Processing Utilities
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

class MovingAverage:
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

class HampelFilter:
    """Filtro robusto para eliminar outliers impulsivos en la frecuencia estimada."""
    def __init__(self, win: int = 7, n_sigmas: float = 3.0):
        self.win = win if win % 2 != 0 else win + 1
        self.k = n_sigmas
        self.buf: Deque[float] = deque(maxlen=self.win)

    def reset(self) -> None: self.buf.clear()

    def step(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) < self.win: return x
        arr = np.array(self.buf)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        sigma = 1.4826 * mad + 1e-6
        if abs(x - median) > self.k * sigma:
            return median
        return x

# ============================================================
# TKEO Core - DESA-1 Algorithm
# ============================================================

class TeagerKaiserEngine:
    """
    Implementación del Algoritmo de Separación de Energía DESA-1.
    Proporciona una estimación casi instantánea de la frecuencia.
    """
    def __init__(self, fs_hz: float):
        self.fs = fs_hz
        self.dt = 1.0 / fs_hz
        self.reset()

    def reset(self) -> None:
        self.x = deque([0.0, 0.0, 0.0], maxlen=3) # x[n], x[n-1], x[n-2]

    @staticmethod
    def psi(x_n, x_m1, x_p1) -> float:
        """Operador Ψ(x) = x[n]^2 - x[n-1]x[n+1]"""
        return x_n**2 - x_m1 * x_p1

    def step(self, sample: float) -> float:
        self.x.append(sample)
        if len(self.x) < 3: return 0.0
        
        # Extraemos muestras para DESA-1
        # x_n es la muestra central para la cual calculamos la energía
        x_n = self.x[1]
        x_m1 = self.x[2] # n-1 (en el deque la derecha es más reciente)
        x_p1 = self.x[0] # n+1
        
        # Algoritmo DESA-1: Usa la diferencia y(n) = x(n) - x(n-1)
        # para una estimación más robusta a cambios de amplitud
        y_n = x_n - x_p1
        
        psi_x = self.psi(x_n, x_p1, x_m1)
        psi_y = self.psi(y_n, self.x[1]-self.x[0], self.x[2]-self.x[1]) # psi(x_n - x_n-1)
        
        # Frecuencia digital de DESA-1: cos(Omega) = 1 - [Psi(y_n) / 2*Psi(x_n)]
        try:
            arg = 1.0 - (psi_y / (2.0 * psi_x + 1e-9))
            omega = np.arccos(_clamp(arg, -1.0, 1.0))
            return omega * self.fs / (2.0 * np.pi)
        except:
            return 0.0

# ============================================================
# Wrapper Estimator
# ============================================================

class TeagerEstimator(BaseEstimator):
    """
    Estimador de Teager-Kaiser con filtrado Hampel y DESA-1.
    Ideal para detectar cambios ultra-rápidos de frecuencia.
    """
    NAME = "Teager"
    FAMILY = "Nonlinear-Energy"

    def reset(self) -> None:
        p = self._params
        self.fs = float(p["fs_hz"])
        self.f0 = float(p.get("f0_hz", 60.0))
        
        self.engine = TeagerKaiserEngine(self.fs)
        self.hampel = HampelFilter(win=int(p.get("hampel_win", 11)))
        self.ma = MovingAverage(win=int(p.get("win_smooth", 20)))

    def step(self, v_sample: float) -> float:
        # 1. Estimación instantánea DESA-1
        f_raw = self.engine.step(v_sample)
        if f_raw == 0.0: f_raw = self.f0
        
        # 2. Filtrado robusto de outliers
        f_clean = self.hampel.step(f_raw)
        
        # 3. Suavizado final
        f_out = self.ma.step(f_clean)
        
        return _clamp(f_out, 40.0, 80.0)

    @property
    def latency_samples(self) -> int:
        # TKEO tiene solo 2 muestras de retardo estructural + el promedio móvil
        return int(2 + 0.5 * self._params.get("win_smooth", 20))