# scenarios/s1_synthetic/frequency_ramp_step.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frequency_ramp_step(
    f0: float = 60.0,
    f_step: float = 59.5,
    t_step: float = 2.0,
    t_back: float = 4.0,
    duration: float = 6.0,
    fs: int = 5000,
    rocof: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a sinusoidal waveform with a smooth frequency disturbance:
    ramp from f0 to f_step at a given RoCoF, hold, then ramp back.

    Returns
    -------
    signal : NDArray[np.float64]
        Generated sinusoidal signal.
    f : NDArray[np.float64]
        Instantaneous frequency array (Hz).
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")

    dt = 1.0 / float(fs)
    t: NDArray[np.float64] = np.arange(0.0, duration, dt, dtype=float)
    f: NDArray[np.float64] = np.full_like(t, f0, dtype=float)

    # Time needed to complete each ramp (avoid div-by-zero if rocof==0)
    if rocof > 0.0:
        ramp_time = float(abs(f_step - f0) / rocof)
    else:
        ramp_time = 0.0

    # Scalar logic kept; could vectorizarse si se desea
    for i, ti in enumerate(t):
        if ti < t_step:
            f[i] = f0
        elif t_step <= ti < t_step + ramp_time:
            f[i] = f0 + np.sign(f_step - f0) * rocof * (ti - t_step)
        elif t_step + ramp_time <= ti < t_back:
            f[i] = f_step
        elif t_back <= ti < t_back + ramp_time:
            f[i] = f_step - np.sign(f_step - f0) * rocof * (ti - t_back)
        else:
            f[i] = f0

    theta: NDArray[np.float64] = (2.0 * np.pi * np.cumsum(f) / float(fs)).astype(
        np.float64, copy=False
    )
    signal: NDArray[np.float64] = np.sin(theta).astype(np.float64, copy=False)
    return signal, f
