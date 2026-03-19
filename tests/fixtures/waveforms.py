"""Waveform fixtures for tests."""
from __future__ import annotations

import numpy as np


def make_pure_sine(f_hz: float = 60.0, fs_hz: float = 10_000.0, T_s: float = 1.0, seed: int = 0) -> tuple:
    """Return (t, v, f_true) arrays for a pure sine at f_hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_s, 1.0 / fs_hz)
    v = np.sin(2 * np.pi * f_hz * t)
    f_true = np.full_like(t, f_hz)
    return t, v, f_true
