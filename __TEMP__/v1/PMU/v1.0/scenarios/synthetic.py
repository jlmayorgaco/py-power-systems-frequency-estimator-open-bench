# scenarios/synthetic.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any


def _base_time(fs: float = 10_000, T: float = 50.0):
    t = np.arange(0, T, 1.0 / fs)
    return t


def scenario_step(
    f0: float = 60.0,
    f1: float = 59.5,
    t_step: float = 2.0,
    fs: float = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    t = _base_time(fs)
    f = np.full_like(t, f0)
    f[t >= t_step] = f1

    v = np.sin(2 * np.pi * np.cumsum(f) / fs)

    meta = {
        "description": "Frequency step event",
        "f0": f0,
        "f1": f1,
        "t_step": t_step,
        "fs": fs,
    }
    return t, v, f, meta


def scenario_ramp(
    f0: float = 60.0,
    rocof: float = -0.5,
    T: float = 5.0,
    fs: float = 10_000,
):
    t = _base_time(fs, T)
    f = f0 + rocof * t
    v = np.sin(2 * np.pi * np.cumsum(f) / fs)

    meta = {
        "description": "Linear frequency ramp",
        "f0": f0,
        "rocof": rocof,
        "fs": fs,
    }
    return t, v, f, meta


def scenario_rocof(
    f0: float = 60.0,
    rocof: float = -2.0,
    T: float = 3.0,
    fs: float = 10_000,
):
    t = _base_time(fs, T)
    f = f0 + rocof * t
    v = np.sin(2 * np.pi * np.cumsum(f) / fs)

    meta = {
        "description": "High RoCoF (IBR-dominated)",
        "f0": f0,
        "rocof": rocof,
        "fs": fs,
    }
    return t, v, f, meta
