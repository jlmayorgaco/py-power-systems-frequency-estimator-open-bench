"""Pure sinusoid generation and phase integration."""

from __future__ import annotations

import numpy as np


def make_sine(
    t: np.ndarray, f_hz: float, A: float = 1.0, phi0_rad: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    phi = phi0_rad + 2.0 * np.pi * f_hz * t
    return A * np.sin(phi), phi
