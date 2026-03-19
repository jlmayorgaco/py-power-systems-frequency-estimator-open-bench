# scenarios/s1_synthetic/frequency_ramp_step.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RampStepConfig:
    """Configuration for a frequency ramp–hold–ramp-back disturbance."""

    f0: float = 60.0
    f_step: float = 59.5
    t_step: float = 2.0
    t_back: float = 4.0


# Module-level default (OK for B008)
DEFAULT_RAMP_STEP_CFG = RampStepConfig()


def frequency_ramp_step(
    cfg: RampStepConfig | None = None,
    duration: float = 6.0,
    fs: int = 5000,
    rocof: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a sinusoidal waveform with a smooth frequency disturbance:
    ramp from cfg.f0 to cfg.f_step at a given RoCoF, hold, then ramp back.

    Parameters
    ----------
    cfg : RampStepConfig | None
        f0, f_step, t_step (ramp start), t_back (ramp-back start).
        If None, uses DEFAULT_RAMP_STEP_CFG.
    duration : float
        Total signal length in seconds. Must be >= 0.
    fs : int
        Sampling rate (Hz). Must be > 0.
    rocof : float
        Rate of change of frequency during ramps (Hz/s). If 0, ramps are skipped.

    Returns
    -------
    signal : NDArray[np.float64]
        Generated sinusoidal signal.
    f : NDArray[np.float64]
        Instantaneous frequency array (Hz).
    """
    cfg = DEFAULT_RAMP_STEP_CFG if cfg is None else cfg

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")

    dt = 1.0 / float(fs)
    t: NDArray[np.float64] = np.arange(0.0, duration, dt, dtype=float)
    f: NDArray[np.float64] = np.full_like(t, cfg.f0, dtype=float)

    # Time needed to complete each ramp (avoid div-by-zero if rocof == 0)
    ramp_time = float(abs(cfg.f_step - cfg.f0) / rocof) if rocof > 0.0 else 0.0
    sgn = 1.0 if (cfg.f_step - cfg.f0) >= 0.0 else -1.0

    # Scalar logic kept; can be vectorized later if needed
    for i, ti in enumerate(t):
        if ti < cfg.t_step:
            f[i] = cfg.f0
        elif cfg.t_step <= ti < cfg.t_step + ramp_time:
            f[i] = cfg.f0 + sgn * rocof * (ti - cfg.t_step)
        elif cfg.t_step + ramp_time <= ti < cfg.t_back:
            f[i] = cfg.f_step
        elif cfg.t_back <= ti < cfg.t_back + ramp_time:
            f[i] = cfg.f_step - sgn * rocof * (ti - cfg.t_back)
        else:
            f[i] = cfg.f0

    theta: NDArray[np.float64] = (2.0 * np.pi * np.cumsum(f) / float(fs)).astype(
        np.float64, copy=False
    )
    signal: NDArray[np.float64] = np.sin(theta).astype(np.float64, copy=False)
    return signal, f
