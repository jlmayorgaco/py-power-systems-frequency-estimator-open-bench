from __future__ import annotations

from collections import deque
from typing import Tuple
import numpy as np
import math  # Necesario para atan2

from .base import BaseEstimator

# ============================================================
# Core Utilities & Transforms
# ============================================================

_EPS = 1e-12


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _wrap_2pi(theta: float) -> float:
    return float(theta) % (2.0 * np.pi)


def _safe_div(num: float, den: float, eps: float = _EPS) -> float:
    d = float(den)
    if not np.isfinite(d) or abs(d) < eps:
        d = eps if d >= 0 else -eps
    return float(num) / d


def _park(v_alpha: float, v_beta: float, theta: float) -> Tuple[float, float]:
    """
    Transformada de Park (alpha-beta -> d-q).
    """
    c, s = np.cos(theta), np.sin(theta)
    v_d = c * v_alpha + s * v_beta
    v_q = -s * v_alpha + c * v_beta
    return float(v_d), float(v_q)


def _inv_park(v_d: float, v_q: float, theta: float) -> Tuple[float, float]:
    """
    Transformada Inversa de Park (d-q -> alpha-beta).
    """
    c, s = np.cos(theta), np.sin(theta)
    v_alpha = c * v_d - s * v_q
    v_beta = s * v_d + c * v_q
    return float(v_alpha), float(v_beta)


class LowPassFilter:
    """Filtro de primer orden simple para la red de desacople."""

    def __init__(self, fs_hz: float, fc_hz: float):
        self.ts = 1.0 / fs_hz
        tau = 1.0 / (2.0 * np.pi * fc_hz)
        self.alpha = self.ts / (tau + self.ts)
        self.y_prev = 0.0

    def reset(self):
        self.y_prev = 0.0

    def step(self, x: float) -> float:
        y = (1.0 - self.alpha) * self.y_prev + self.alpha * x
        self.y_prev = y
        return float(y)


# ============================================================
# SOGI-QSG (Trapezoidal / Tustin integration, stable)
# ============================================================


class SOGI_QSG:
    """
    Quadrature Signal Generator (SOGI).
    Genera (v_alpha, v_beta) a partir de una señal escalar v_in.
    """

    def __init__(self, fs_hz: float, f0_hz: float, k: float = 1.414):
        self.fs = float(fs_hz)
        self.dt = 1.0 / self.fs
        self.k = float(k)
        self.w0 = 2.0 * np.pi * float(f0_hz)
        self.reset()

    def reset(self) -> None:
        self.va = 0.0
        self.vb = 0.0
        self.va_dot_prev = 0.0
        self.vb_dot_prev = 0.0

    def step(self, v_in: float, w_est: float) -> Tuple[float, float]:
        w = float(w_est)
        if not np.isfinite(w):
            w = self.w0
        w = _clamp(w, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0)

        err = float(v_in - self.va)
        va_dot = (self.k * w * err) - (w * self.vb)
        vb_dot = w * self.va

        va_new = self.va + 0.5 * self.dt * (self.va_dot_prev + va_dot)
        vb_new = self.vb + 0.5 * self.dt * (self.vb_dot_prev + vb_dot)

        self.va, self.vb = float(va_new), float(vb_new)
        self.va_dot_prev, self.vb_dot_prev = float(va_dot), float(vb_dot)

        if not (np.isfinite(self.va) and np.isfinite(self.vb)):
            self.reset()
            return 0.0, 0.0

        return self.va, self.vb


# ============================================================
# 1) SRF-PLL (Synchronous Reference Frame)
# ============================================================


class SRF_PLL:
    def __init__(
        self, fs_hz: float, f0_hz: float, kp: float, ki: float, sogi_k: float = 1.414
    ):
        self.fs = float(fs_hz)
        self.dt = 1.0 / self.fs
        self.f0 = float(f0_hz)
        self.kp = float(kp)
        self.ki = float(ki)
        self.qsg = SOGI_QSG(fs_hz, f0_hz, sogi_k)
        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.f_hat = float(self.f0)
        self.int_e = 0.0
        self.int_e_lim = 10.0  # Limite anti-windup
        self.v_eps = 1e-6
        self.qsg.reset()

    def step(self, v: float) -> float:
        va, vb = self.qsg.step(float(v), 2.0 * np.pi * float(self.f_hat))
        v_d, v_q = _park(va, vb, self.theta)

        # [CORRECCIÓN Q1]: Usar atan2(vq, vd) en lugar de vq/vd.
        # Esto es mucho más robusto para errores de fase grandes (ej. arranque).
        e = math.atan2(v_q, v_d)

        self.int_e = _clamp(self.int_e + e * self.dt, -self.int_e_lim, self.int_e_lim)
        f_delta = self.kp * e + self.ki * self.int_e
        f_new = _clamp(self.f0 + f_delta, 40.0, 80.0)

        if not np.isfinite(f_new):
            f_new = float(self.f0)

        self.f_hat = float(f_new)
        self.theta = _wrap_2pi(self.theta + (2.0 * np.pi * self.f_hat) * self.dt)
        return float(self.f_hat)


# ============================================================
# 2) MAF-SRF-PLL (Moving Average Filter)
# ============================================================


class MAF_SRF_PLL(SRF_PLL):
    def __init__(
        self,
        fs_hz: float,
        f0_hz: float,
        kp: float,
        ki: float,
        win_len: int,
        sogi_k: float = 1.414,
    ):
        self.win_len = int(max(1, win_len))
        self.maf = deque(maxlen=self.win_len)
        super().__init__(fs_hz, f0_hz, kp, ki, sogi_k=sogi_k)

    def reset(self) -> None:
        super().reset()
        self.maf.clear()

    def step(self, v: float) -> float:
        va, vb = self.qsg.step(float(v), 2.0 * np.pi * float(self.f_hat))
        v_d, v_q = _park(va, vb, self.theta)

        # [CORRECCIÓN Q1]: Usar atan2(vq, vd).
        # Crucial aquí porque el MAF puede introducir retardos que lleven
        # la fase temporalmente cerca de 90 grados en transitorios.
        e_inst = math.atan2(v_q, v_d)

        self.maf.append(e_inst)
        e = float(sum(self.maf) / max(1, len(self.maf)))

        self.int_e = _clamp(self.int_e + e * self.dt, -self.int_e_lim, self.int_e_lim)
        f_delta = self.kp * e + self.ki * self.int_e
        f_new = _clamp(self.f0 + f_delta, 40.0, 80.0)

        if not np.isfinite(f_new):
            f_new = float(self.f0)

        self.f_hat = float(f_new)
        self.theta = _wrap_2pi(self.theta + (2.0 * np.pi * self.f_hat) * self.dt)
        return float(self.f_hat)


# ============================================================
# 3) DDSRF-PLL (Decoupled Double Synchronous Reference Frame)
# ============================================================


class DDSRF_PLL:
    """
    Implementación REAL del DDSRF-PLL.
    Usa dos marcos de referencia (Positivo y Negativo) y una red de desacople
    para cancelar la oscilación de 2*omega causada por desbalance.
    """

    def __init__(
        self, fs_hz: float, f0_hz: float, kp: float, ki: float, sogi_k: float = 1.414
    ):
        self.fs = float(fs_hz)
        self.dt = 1.0 / self.fs
        self.f0 = float(f0_hz)

        # Controlador PI para el marco positivo
        self.kp = float(kp)
        self.ki = float(ki)

        # Generador SOGI para obtener Alpha-Beta
        self.qsg = SOGI_QSG(fs_hz, f0_hz, sogi_k)

        # Filtros de desacople (Low Pass Filters)
        # Frecuencia de corte tipica: omega_grid / sqrt(2) para mejor respuesta
        fc_decouple = f0_hz / 1.414
        self.lpf_d_pos = LowPassFilter(fs_hz, fc_decouple)
        self.lpf_q_pos = LowPassFilter(fs_hz, fc_decouple)
        self.lpf_d_neg = LowPassFilter(fs_hz, fc_decouple)
        self.lpf_q_neg = LowPassFilter(fs_hz, fc_decouple)

        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.f_hat = float(self.f0)
        self.int_e = 0.0
        self.int_e_lim = 10.0
        self.qsg.reset()

        self.lpf_d_pos.reset()
        self.lpf_q_pos.reset()
        self.lpf_d_neg.reset()
        self.lpf_q_neg.reset()

    def step(self, v: float) -> float:
        # 1. Obtener componentes estacionarios (Alpha-Beta)
        w_curr = 2.0 * np.pi * float(self.f_hat)
        va, vb = self.qsg.step(float(v), w_curr)

        # 2. Transformada a ejes dq síncronos rotando a +theta y -theta
        # Eje Positivo (gira con la red)
        vd_p_raw, vq_p_raw = _park(va, vb, self.theta)
        # Eje Negativo (gira contrario)
        vd_n_raw, vq_n_raw = _park(va, vb, -self.theta)

        # 3. Filtrado paso bajo para extraer valores medios (DC components)
        # Estos son los valores "desacoplados" estimados
        vd_p_bar = self.lpf_d_pos.step(vd_p_raw)
        vq_p_bar = self.lpf_q_pos.step(vq_p_raw)
        vd_n_bar = self.lpf_d_neg.step(vd_n_raw)
        vq_n_bar = self.lpf_q_neg.step(vq_n_raw)

        # 4. Red de Desacople (Decoupling Network)
        cos_2theta = np.cos(2.0 * self.theta)
        sin_2theta = np.sin(2.0 * self.theta)

        # Ecuaciones de desacople cruzado
        # Restamos la componente oscilatoria de secuencia negativa del eje positivo
        vq_p_decoupled = vq_p_raw - (vd_n_bar * sin_2theta - vq_n_bar * cos_2theta)

        # 5. Normalización robusta
        mag_p = np.hypot(vd_p_bar, vq_p_bar)
        if mag_p < 1e-6:
            err_norm = 0.0
        else:
            # Usar la señal desacoplada y normalizar
            err_norm = vq_p_decoupled / mag_p

        # 6. Controlador PI
        self.int_e = _clamp(
            self.int_e + err_norm * self.dt, -self.int_e_lim, self.int_e_lim
        )
        f_delta = self.kp * err_norm + self.ki * self.int_e

        f_new = _clamp(self.f0 + f_delta, 40.0, 80.0)
        if not np.isfinite(f_new):
            f_new = float(self.f0)

        self.f_hat = float(f_new)
        self.theta = _wrap_2pi(self.theta + w_curr * self.dt)

        return float(self.f_hat)


# ============================================================
# Wrappers para el Benchmark Framework
# ============================================================


class SRFPLLEstimator(BaseEstimator):
    NAME = "SRF-PLL"
    FAMILY = "PLL"

    def reset(self) -> None:
        p = self._params
        self._impl = SRF_PLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("kp", 20.0),
            p.get("ki", 2000.0),
            p.get("sogi_k", 1.414),
        )

    def _step(self, v: float) -> float:
        return float(self._impl.step(float(v)))

    @property
    def latency_samples(self) -> int:
        return int(self._params.get("latency_samples", 1))


class MAFPLLEstimator(BaseEstimator):
    NAME = "MAF-SRF-PLL"
    FAMILY = "PLL"

    def reset(self) -> None:
        p = self._params
        self.win_maf = int(p.get("win_maf", 166))
        # AJUSTE: Ganancias reducidas para estabilidad con MAF
        self._impl = MAF_SRF_PLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("kp", 1.0),
            p.get("ki", 50.0),
            self.win_maf,
            p.get("sogi_k", 1.414),
        )

    def _step(self, v: float) -> float:
        return float(self._impl.step(float(v)))

    @property
    def latency_samples(self) -> int:
        return int(self.win_maf // 2)


class DDSRFPLLEstimator(BaseEstimator):
    NAME = "DDSRF-PLL"
    FAMILY = "PLL"

    def reset(self) -> None:
        p = self._params
        # DDSRF suele requerir gains un poco más agresivos que SRF simple
        # porque la señal q+ ya viene limpia de oscilaciones 2w.
        self._impl = DDSRF_PLL(
            p["fs_hz"],
            p.get("f0_hz", 60.0),
            p.get("kp", 10.0),
            p.get("ki", 500.0),  # Un poco más lento que SRF para dejar actuar al LPF
            p.get("sogi_k", 1.414),
        )

    def _step(self, v: float) -> float:
        return float(self._impl.step(float(v)))

    @property
    def latency_samples(self) -> int:
        return int(self._params.get("latency_samples", 1))
