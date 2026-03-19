from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np
from .base import BaseEstimator

# ============================================================
# Robust Signal Processing Utilities
# ============================================================

_EPS = 1e-12


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


class MovingAverage:
    def __init__(self, win: int):
        self.win = int(max(1, win))
        self.buf: Deque[float] = deque(maxlen=self.win)
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
        return self.sum / len(self.buf)


class HampelFilter:
    """Filtro robusto para eliminar outliers impulsivos en la frecuencia estimada."""

    def __init__(self, win: int = 7, n_sigmas: float = 3.0):
        self.win = int(win if win % 2 != 0 else win + 1)
        self.k = float(n_sigmas)
        self.buf: Deque[float] = deque(maxlen=self.win)

    def reset(self) -> None:
        self.buf.clear()

    def step(self, x: float) -> float:
        x = float(x)
        self.buf.append(x)
        if len(self.buf) < self.win:
            return x
        arr = np.asarray(self.buf, dtype=float)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        sigma = 1.4826 * mad + 1e-9
        if abs(x - median) > self.k * sigma:
            return median
        return x


# ============================================================
# TKEO Core - DESA-1 Algorithm
# ============================================================


class TeagerKaiserEngine:
    """
    Implementación CAUSAL de DESA-1 con índice correcto.
    Usa 4 muestras para estimar f centrada en x[n-1] (1 muestra de retardo).
    """

    def __init__(self, fs_hz: float):
        self.fs = float(fs_hz)
        self.dt = 1.0 / float(fs_hz)
        self.reset()

    def reset(self) -> None:
        # guardamos x[n-3], x[n-2], x[n-1], x[n] (nuevo entra a la derecha)
        self.x: Deque[float] = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)

    @staticmethod
    def psi(x_c: float, x_prev: float, x_next: float) -> float:
        """Operador Ψ(x) = x_c^2 - x_prev * x_next"""
        return float(x_c * x_c - x_prev * x_next)

    def step(self, sample: float) -> float:
        s = float(sample)
        if not np.isfinite(s):
            return 0.0

        self.x.append(s)
        if len(self.x) < 4:
            return 0.0

        # Desempaquetado consistente:
        # x0=x[n-3], x1=x[n-2], x2=x[n-1], x3=x[n]
        x0, x1, x2, x3 = (
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
            float(self.x[3]),
        )

        # 1) Psi(x) centrada en x[n-1] = x2 con vecinos x1 y x3:
        psi_x = self.psi(x2, x1, x3)

        # Guardrail: si la energía es demasiado baja, no se puede estimar
        if (not np.isfinite(psi_x)) or (abs(psi_x) < 1e-12):
            return 0.0

        # 2) Diferencias y[n] = x[n] - x[n-1]
        # y2 = y[n]   = x3 - x2
        # y1 = y[n-1] = x2 - x1
        # y0 = y[n-2] = x1 - x0
        y2 = x3 - x2
        y1 = x2 - x1
        y0 = x1 - x0

        # Psi(y) centrada en y[n-1] = y1 con vecinos y0,y2:
        psi_y = self.psi(y1, y0, y2)

        if (not np.isfinite(psi_y)) or (psi_y < 0.0):
            # psi_y puede volverse negativa por ruido/outliers → invalida DESA
            return 0.0

        # DESA-1:
        # cos(Ω) = 1 - psi_y / (2*psi_x)
        arg = 1.0 - (psi_y / (2.0 * psi_x + 1e-12))
        arg = _clamp(arg, -1.0, 1.0)

        omega = float(np.arccos(arg))  # [rad/sample]
        f_hz = omega * self.fs / (2.0 * np.pi)

        if not np.isfinite(f_hz):
            return 0.0

        return float(f_hz)


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

        # explícito: resetea buffers internos
        self.engine.reset()
        self.hampel.reset()
        self.ma.reset()

    def _step(self, v_sample: float) -> float:
        # 1) Estimación instantánea DESA-1 (puede devolver 0 si no es fiable aún)
        f_raw = float(self.engine.step(float(v_sample)))
        if (not np.isfinite(f_raw)) or (f_raw <= 0.0):
            f_raw = float(self.f0)

        # 2) Filtrado robusto de outliers
        f_clean = float(self.hampel.step(f_raw))

        # 3) Suavizado final
        f_out = float(self.ma.step(f_clean))

        # 4) límites físicos
        return _clamp(f_out, 40.0, 80.0)

    @property
    def latency_samples(self) -> int:
        # DESA usa 4 muestras → retardo estructural ~1 muestra (centrado en x[n-1])
        # + el smoothing (aprox (N-1)/2)
        win = int(self._params.get("win_smooth", 20))
        return int(1 + (win // 2))
