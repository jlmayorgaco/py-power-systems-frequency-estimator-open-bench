from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple
import numpy as np
from .base import BaseEstimator

# ============================================================
# Core Utilities
# ============================================================

_EPS = 1e-12


def _clamp(x: float, lo: float, hi: float) -> float:
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    return float(min(max(x, lo), hi))


def _safe_div(num: float, den: float, eps: float = _EPS) -> float:
    d = float(den)
    if abs(d) < eps:
        d = eps if d >= 0 else -eps
    return float(num) / d


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
        x = float(x)
        if len(self.buf) == self.win:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        return float(self.sum / max(1, len(self.buf)))


# ============================================================
# 1) SOGI-QSG (Trapezoidal / Tustin)
# ============================================================


class SOGI_QSG:
    """
    Quadrature Signal Generator basado en SOGI.
    Implementa integración trapezoidal (Tustin) con derivada previa.
    """

    def __init__(self, fs_hz: float, k: float = 1.414):
        self.fs = float(fs_hz)
        self.dt = 1.0 / self.fs
        self.k = float(k)
        self.reset()

    def reset(self) -> None:
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.va_dot_old = 0.0
        self.vb_dot_old = 0.0

    def step(self, v_in: float, w: float) -> Tuple[float, float, float]:
        v_in = float(v_in)
        w = float(w)

        # Guardrail: ω físicamente plausible
        w = _clamp(w, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0)

        # Error de entrada
        e = v_in - self.v_alpha

        # Derivadas
        va_dot = self.k * w * e - w * self.v_beta
        vb_dot = w * self.v_alpha

        # Integración trapezoidal
        self.v_alpha += (self.dt / 2.0) * (va_dot + self.va_dot_old)
        self.v_beta += (self.dt / 2.0) * (vb_dot + self.vb_dot_old)

        # Actualiza derivadas previas
        self.va_dot_old = va_dot
        self.vb_dot_old = vb_dot

        # Anti-NaN (fail-soft)
        if not (np.isfinite(self.v_alpha) and np.isfinite(self.v_beta)):
            self.reset()
            return 0.0, 0.0, float(e)

        return float(self.v_alpha), float(self.v_beta), float(e)


# ============================================================
# 2) Classic SOGI-FLL (Standard & Industrial)
# ============================================================


class ClassicSOGIFLL:
    """Lazo de frecuencia basado en SOGI con normalización opcional."""

    def __init__(
        self,
        fs_hz: float,
        f0_hz: float,
        gamma: float,
        k: float,
        use_amp_norm: bool = True,
    ):
        self.fs = float(fs_hz)
        self.f0 = float(f0_hz)
        self.dt = 1.0 / self.fs
        self.gamma = float(gamma)
        self.k = float(k)
        self.use_amp_norm = bool(use_amp_norm)

        self.qsg = SOGI_QSG(self.fs, self.k)
        self.w = 2.0 * np.pi * self.f0

        # Estimador de amplitud cuadrado
        self.A2 = 1.0
        # LPF ~20ms
        self.alpha_amp = (self.dt) / (0.02 + self.dt)

        # Gating por magnitud (sags / señal casi cero)
        self.v_eps = 1e-6

    def reset(self) -> None:
        self.qsg.reset()
        self.w = 2.0 * np.pi * self.f0
        self.A2 = 1.0

    def step(self, v_in: float) -> float:
        v_in = float(v_in)
        va, vb, e = self.qsg.step(v_in, self.w)

        # Magnitud estimada de la componente fundamental
        v_mag = float(np.hypot(va, vb))

        # Estimación de A^2 (LPF sobre va^2 + vb^2)
        inst_a2 = float(va * va + vb * vb)
        self.A2 = (1.0 - self.alpha_amp) * float(self.A2) + self.alpha_amp * inst_a2

        # Si la señal colapsa: congela adaptación (evita que ω “huya”)
        if (not np.isfinite(v_mag)) or (v_mag < self.v_eps):
            return float(self.w / (2.0 * np.pi))

        # Ley de adaptación FLL
        denom = (float(self.A2) + 1e-6) if self.use_amp_norm else 1.0
        dw = -self.gamma * float(e) * float(vb) / denom

        # Integración + límites de ω
        self.w = _clamp(self.w + self.dt * dw, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0)

        # Anti-NaN
        if not np.isfinite(self.w):
            self.w = 2.0 * np.pi * self.f0

        return float(self.w / (2.0 * np.pi))


# ============================================================
# 3) MSOGI-FLL (Parallel Harmonic Rejection)
# ============================================================


class MSOGIFLL:
    """Multi-Resonant SOGI para rechazo de armónicos."""

    def __init__(
        self, fs_hz: float, f0_hz: float, gamma: float, harmonics: Sequence[int]
    ):
        self.fs = float(fs_hz)
        self.f0 = float(f0_hz)
        self.gamma = float(gamma)

        self.fund = ClassicSOGIFLL(self.fs, self.f0, self.gamma, 1.414, True)
        self.h_orders = tuple(int(h) for h in harmonics)
        self.h_qsgs = [SOGI_QSG(self.fs, k=1.0) for _ in self.h_orders]

    def reset(self) -> None:
        self.fund.reset()
        for qsg in self.h_qsgs:
            qsg.reset()

    def step(self, v_in: float) -> float:
        v_in = float(v_in)

        # Usa ω del fundamental como referencia
        w_est = float(self.fund.w)
        w_est = _clamp(w_est, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0)

        # Cancelación en paralelo: estima armónicos y los resta
        v_harm = 0.0
        for h, qsg in zip(self.h_orders, self.h_qsgs):
            w_h = float(h) * w_est
            # clamp para armónicos (aun así evita w enorme si h grande)
            w_h = _clamp(
                w_h, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0 * max(1.0, float(h))
            )
            va_h, _, _ = qsg.step(v_in, w_h)
            v_harm += float(va_h)

        return float(self.fund.step(v_in - v_harm))


# ============================================================
# 4) Estimator Wrappers (Compatibles con BaseEstimator nuevo)
#   - Implementan _step()
#   - NO pisan step()
# ============================================================


class SOGIClassicEstimator(BaseEstimator):
    NAME = "SOGI-Classic"
    FAMILY = "SOGI-FLL"

    def reset(self) -> None:
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = ClassicSOGIFLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("g", 100.0),
            p.get("k", 1.414),
            False,  # classic sin amp norm (como tenías)
        )

    def _step(self, v: float) -> float:
        return float(self._ma.step(self._impl.step(float(v))))

    @property
    def latency_samples(self) -> int:
        # Esto era una heurística tuya. Lo dejo, pero robusto a falta de fs_hz:
        fs = float(self._params.get("fs_hz", 0.0))
        if fs <= 0:
            return 0
        return int(0.8 * (fs / 60.0))


class SOGIIndustrialEstimator(BaseEstimator):
    NAME = "SOGI-Industrial"
    FAMILY = "SOGI-FLL"

    def reset(self) -> None:
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = ClassicSOGIFLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("g", 100.0),
            p.get("k", 1.414),
            True,  # industrial con amp norm (como tenías)
        )

    def _step(self, v: float) -> float:
        return float(self._ma.step(self._impl.step(float(v))))

    @property
    def latency_samples(self) -> int:
        fs = float(self._params.get("fs_hz", 0.0))
        if fs <= 0:
            return 0
        return int(0.9 * (fs / 60.0))


class MSOGIFLLEstimator(BaseEstimator):
    NAME = "MSOGI-FLL"
    FAMILY = "SOGI-FLL"

    def reset(self) -> None:
        p = self._params
        self._ma = MovingAverage(p.get("smooth_win", 10))
        self._impl = MSOGIFLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("g", 100.0),
            p.get("harmonics", (3, 5, 7)),
        )

    def _step(self, v: float) -> float:
        return float(self._ma.step(self._impl.step(float(v))))

    @property
    def latency_samples(self) -> int:
        fs = float(self._params.get("fs_hz", 0.0))
        if fs <= 0:
            return 0
        return int(1.1 * (fs / 60.0))
