# scenarios/s1_synthetic/make_clean.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_clean(
    f0: float = 60.0,
    df: float = 0.0,
    duration: float = 5.0,
    fs: int = 5000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Genera una sinusoide 'limpia' con rampa de frecuencia opcional.

    Parámetros
    ----------
    f0 : float
        Frecuencia inicial [Hz].
    df : float
        Cambio total de frecuencia a lo largo de `duration` [Hz].
    duration : float
        Duración de la señal [s].
    fs : int
        Frecuencia de muestreo [Hz].

    Returns
    -------
    signal : NDArray[np.float64]
        Señal senoidal.
    f : NDArray[np.float64]
        Frecuencia instantánea (Hz) para cada muestra.
    """
    dt = 1.0 / float(fs)
    t: NDArray[np.float64] = np.arange(0.0, duration, dt, dtype=float)
    if duration > 0.0:
        f: NDArray[np.float64] = (f0 + df * t / duration).astype(np.float64, copy=False)
    else:
        f = np.full_like(t, float(f0), dtype=np.float64)

    theta: NDArray[np.float64] = (2.0 * np.pi * np.cumsum(f) / float(fs)).astype(
        np.float64, copy=False
    )
    signal: NDArray[np.float64] = np.sin(theta).astype(np.float64, copy=False)
    return signal, f
