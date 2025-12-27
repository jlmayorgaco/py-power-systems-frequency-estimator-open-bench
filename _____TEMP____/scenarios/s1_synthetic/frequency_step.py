from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def frequency_step(
    f0: float = 60.0,
    f_step: float = 59.5,
    t_step: float = 2.0,
    t_back: float = 4.0,
    duration: float = 6.0,
    fs: float = 5000.0,
) -> tuple[FloatArray, FloatArray]:
    """
    Generate a sinusoidal waveform with a step change in frequency.

    Parameters
    ----------
    f0 : float
        Nominal frequency (Hz).
    f_step : float
        Frequency during disturbance (Hz).
    t_step : float
        Time (s) when frequency steps from f0 to f_step.
    t_back : float
        Time (s) when frequency returns to f0.
    duration : float
        Total duration (s).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    signal : FloatArray
        Generated sinusoidal signal (shape: [N]).
    f : FloatArray
        Instantaneous frequency array (Hz) (shape: [N]).
    """
    fs = float(fs)

    # Time base (half-open interval [0, duration))
    t: FloatArray = np.arange(0.0, float(duration), 1.0 / fs, dtype=float)

    # Piecewise frequency profile
    f: FloatArray = np.full_like(t, float(f0), dtype=float)
    step_mask = (t >= float(t_step)) & (t < float(t_back))
    f[step_mask] = float(f_step)

    # Phase accumulation and signal
    theta: FloatArray = (2.0 * np.pi * np.cumsum(f) / fs).astype(np.float64, copy=False)
    signal: FloatArray = np.sin(theta).astype(np.float64, copy=False)

    return signal, f
