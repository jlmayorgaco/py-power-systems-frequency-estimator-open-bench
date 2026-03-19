from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from .base import BaseEstimator


@dataclass(frozen=True)
class IpDFTConfig:
    """
    Configuración técnica para Interpolated DFT.
    Basado en el formalismo de Jain para estimación espectral de alta resolución.
    """

    fs_hz: float
    f0_hz: float = 60.0
    cycles: int = 12  # Ventana de observación (trade-off: resolución vs latencia)
    decim: int = 1  # Intervalo de actualización computacional
    remove_dc: bool = True  # Filtrado de componente de continua
    window_type: str = "hann"


class IpDFTEstimator:
    """
    Interpolated DFT (IpDFT) con ventana de Hann e interpolación de Jain.
    Optimizado para seguimiento de frecuencia en redes eléctricas con distorsión.
    """

    def __init__(self, cfg: IpDFTConfig):
        self.cfg = cfg
        self.dt_update = cfg.decim / cfg.fs_hz

        # N calculado para ciclos exactos a f_nominal para minimizar filtración
        self.N = int(round((cfg.fs_hz / cfg.f0_hz) * cfg.cycles))
        self.N = max(self.N, 32)

        self.buf = deque(maxlen=self.N)

        # Ventana de Hann Periódica: Estándar para IpDFT (máximo rechazo de lóbulos)
        n = np.arange(self.N)
        self.win = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / self.N))

        # Ganancia de recuperación de la ventana (Coherent Gain)
        self.win_gain = np.sum(self.win) / self.N
        self.res_hz = cfg.fs_hz / self.N

        self._last_f = cfg.f0_hz
        self._count = 0

    def reset(self) -> None:
        self.buf.clear()
        self._last_f = self.cfg.f0_hz
        self._count = 0

    def step(self, z: float) -> float:
        self._count += 1
        self.buf.append(float(z))

        # Esperar a tener el buffer lleno
        if len(self.buf) < self.N:
            return self.cfg.f0_hz

        # Actualización diezmada para eficiencia en DSP
        if (self._count % self.cfg.decim) != 0:
            return self._last_f

        # 1. Preparación de señal
        x = np.array(self.buf)
        if self.cfg.remove_dc:
            x -= np.mean(x)

        # 2. Ventaneo y FFT Real
        xw = x * self.win
        X = np.fft.rfft(xw)
        mags = np.abs(X)

        # 3. Búsqueda del pico espectral (restringida a banda f_nom +/- 10Hz)
        # Esto evita "lock-on" en armónicos durante el benchmark
        k_min = int(max(1, (self.cfg.f0_hz - 10) / self.res_hz))
        k_max = int(min(len(mags) - 2, (self.cfg.f0_hz + 10) / self.res_hz))
        k = k_min + np.argmax(mags[k_min : k_max + 1])

        # 4. Interpolación de Jain (Jain's Method para Hann)
        # Ref: "The use of Hann window in digital spectrum analysis"
        y_k = mags[k]

        # Determinar dirección del lóbulo adyacente
        if mags[k + 1] > mags[k - 1]:
            y_adj = mags[k + 1]
            sign = 1
        else:
            y_adj = mags[k - 1]
            sign = -1

        # Estimación del desplazamiento fraccional (delta)
        # Fórmula analítica para Hann: delta = sign * (2 * R - 1) / (R + 1)
        # donde R = y_adj / y_k
        R = y_adj / (y_k + 1e-18)
        delta = sign * (2.0 * R - 1.0) / (R + 1.0)
        delta = np.clip(delta, -0.5, 0.5)

        f_hat = (k + delta) * self.res_hz

        self._last_f = float(f_hat)
        return self._last_f


class IpDFT(BaseEstimator):
    """
    Interface Wrapper para el Benchmark Framework.
    NAME: IpDFT
    FAMILY: Fourier
    """

    NAME = "IpDFT"
    FAMILY = "Fourier"

    def reset(self) -> None:
        # Recuperación de parámetros del Diccionario de Tuning
        fs = float(self._params.get("fs_dsp_hz", self._params.get("fs_hz", 10000.0)))
        f0 = float(self._params.get("f0_hz", 60.0))
        c = int(self._params.get("window_cycles", self._params.get("cycles", 12)))
        d = int(self._params.get("decim", 1))

        cfg = IpDFTConfig(fs_hz=fs, f0_hz=f0, cycles=c, decim=d, remove_dc=True)
        self._impl = IpDFTEstimator(cfg)

    def step(self, v_sample: float) -> float:
        return self._impl.step(v_sample)

    @property
    def latency_samples(self) -> int:
        """
        Retraso de grupo (Group Delay) para ventana simétrica de Hann.
        En Fourier, el punto de estimación es el centro de la ventana.
        """
        fs = float(self._params.get("fs_dsp_hz", self._params.get("fs_hz", 10000.0)))
        f0 = float(self._params.get("f0_hz", 60.0))
        c = int(self._params.get("window_cycles", self._params.get("cycles", 12)))

        N = int(round((fs / f0) * c))
        return N // 2
