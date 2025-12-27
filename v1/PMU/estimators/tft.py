from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np
from .base import BaseEstimator

# ============================================================
# High-Precision Utilities
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

# ============================================================
# STFT-based Estimator (TFT-Classic)
# ============================================================

class TFTClassic:
    """
    Estimador de Frecuencia mediante STFT e interpolación log-parabólica.
    Proporciona un balance robusto entre resolución y latencia.
    """
    def __init__(self, fs_hz: float, win_sec: float, hop_sec: float):
        self.fs = fs_hz
        self.N = int(round(win_sec * fs_hz))
        self.H = int(round(hop_sec * fs_hz))
        
        # Ventana de Hann para suprimir lóbulos laterales (Side-lobes)
        self.win = np.hanning(self.N)
        self.buf = deque(maxlen=self.N)
        
        self.res_hz = fs_hz / self.N
        self._last_f = 60.0
        self._count = 0

    def reset(self) -> None:
        self.buf.clear()
        self._count = 0

    def _get_interpolated_freq(self, x: np.ndarray) -> float:
        # FFT Real con ventaneo
        X = np.fft.rfft(x * self.win)
        mags = np.abs(X)
        
        # Búsqueda de pico en banda de interés (40-80 Hz)
        k_min = int(40 / self.res_hz)
        k_max = int(80 / self.res_hz)
        k = k_min + np.argmax(mags[k_min:k_max+1])
        
        # Interpolación parabólica en dominio LOG para mayor precisión
        # Ref: "Parabolic Interpolation of Spectral Peaks"
        try:
            l_m1 = np.log(mags[k-1] + 1e-12)
            l_0  = np.log(mags[k] + 1e-12)
            l_p1 = np.log(mags[k+1] + 1e-12)
            
            delta = 0.5 * (l_m1 - l_p1) / (l_m1 - 2*l_0 + l_p1 + 1e-12)
            return (k + delta) * self.res_hz
        except:
            return k * self.res_hz

    def step(self, v_in: float) -> float:
        self.buf.append(v_in)
        self._count += 1
        
        if len(self.buf) < self.N: return 60.0
        
        # Actualizar solo cada 'hop' (salto de ventana)
        if self._count % self.H == 0:
            f_meas = self._get_interpolated_freq(np.array(self.buf))
            self._last_f = _clamp(f_meas, 40.0, 80.0)
            
        return self._last_f

# ============================================================
# SDFT-based Estimator (TFT-SOTA)
# ============================================================

class TFTSDFT:
    """
    Sliding DFT (SDFT) con factor de amortiguamiento para estabilidad.
    Permite una actualización muestra a muestra con bajo costo computacional.
    """
    def __init__(self, fs_hz: float, f0_hz: float, N_cycles: int = 4):
        self.fs = fs_hz
        self.N = int(round(N_cycles * (fs_hz / f0_hz)))
        self.beta = 0.9999  # Factor de estabilidad para evitar deriva numérica
        
        # Frecuencias de interés (bins) centradas en f0
        self.k_fund = int(round(f0_hz * self.N / fs_hz))
        self.bins = [self.k_fund - 1, self.k_fund, self.k_fund + 1]
        
        # Coeficientes de rotación compleja
        self.twiddles = np.exp(2j * np.pi * np.array(self.bins) / self.N)
        self.X = np.zeros(3, dtype=complex)
        self.delay_line = deque([0.0] * self.N, maxlen=self.N)
        
        self.res_hz = fs_hz / self.N

    def reset(self) -> None:
        self.X.fill(0j)
        self.delay_line = deque([0.0] * self.N, maxlen=self.N)

    def step(self, x_n: float) -> float:
        x_old = self.delay_line[0]
        self.delay_line.append(x_n)
        
        # Actualización recursiva del bin: X[k] = (X[k] + x_n - x_old) * e^(j2πk/N)
        # Aplicamos beta para estabilidad
        delta = x_n - (self.beta**self.N) * x_old
        self.X = (self.X + delta) * self.twiddles
        
        # Interpolación de Jain sobre los 3 bins para frecuencia exacta
        mags = np.abs(self.X)
        k_peak = 1 # El bin central es nuestro objetivo
        
        # Interpolación rápida
        if mags[2] > mags[0]:
            r = mags[2] / mags[1]
            delta = r / (1 + r)
        else:
            r = mags[0] / mags[1]
            delta = -r / (1 + r)
            
        f_hat = (self.bins[1] + delta) * self.res_hz
        return _clamp(f_hat, 40.0, 80.0)

# ============================================================
# Wrapper Estimator
# ============================================================

class TFTEstimator(BaseEstimator):
    NAME = "TFT"
    FAMILY = "Time-Frequency"

    def reset(self) -> None:
        p = self._params
        variant = str(p.get("variant", "classic")).lower()
        fs = float(p["fs_hz"])
        
        if variant == "classic":
            self._impl = TFTClassic(fs, p.get("win_sec", 0.08), p.get("hop_sec", 0.01))
        else:
            self._impl = TFTSDFT(fs, p.get("f0_hz", 60.0))
            
        self._ma = MovingAverage(int(p.get("smooth_win", 5)))

    def step(self, v_sample: float) -> float:
        return self._ma.step(self._impl.step(v_sample))

    @property
    def latency_samples(self) -> int:
        # El retraso de grupo es N/2 para métodos de ventana
        N = getattr(self._impl, "N", 800)
        return int(N // 2)