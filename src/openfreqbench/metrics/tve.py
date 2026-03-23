"""
openfreqbench/metrics/tve.py

Total Vector Error (TVE) module for IEEE C37.118.1 compliance.
Calculates the vectorial distance between the theoretical phasor and estimated phasor.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any

from openfreqbench.metrics.frequency import _to_1d, _finite_1d, rmse

def compute_tve_array(
    mag_est: Any,
    phase_est_rad: Any,
    mag_true: Any,
    phase_true_rad: Any
) -> np.ndarray:
    """
    Computes TVE line trace per IEEE C37.118.
    TVE = sqrt( (X_r - X_r_t)^2 + (X_i - X_i_t)^2 ) / |X_t|
    """
    me = _to_1d(mag_est)
    pe = _to_1d(phase_est_rad)
    mt = _to_1d(mag_true)
    pt = _to_1d(phase_true_rad)
    
    n = min(len(me), len(pe), len(mt), len(pt))
    if n == 0:
        return np.array([])
        
    me, pe, mt, pt = me[:n], pe[:n], mt[:n], pt[:n]
    
    # Avoid div by zero
    mt_safe = np.where(mt < 1e-9, 1e-9, mt)
    
    # Real and Imag parts
    Xr_e = me * np.cos(pe)
    Xi_e = me * np.sin(pe)
    
    Xr_t = mt * np.cos(pt)
    Xi_t = mt * np.sin(pt)
    
    num2 = (Xr_e - Xr_t)**2 + (Xi_e - Xi_t)**2
    den2 = Xr_t**2 + Xi_t**2
    den2 = np.where(den2 < 1e-18, 1e-18, den2)
    
    tve_pct = np.sqrt(num2 / den2) * 100.0
    return tve_pct

def tve_max(tve_array: Any) -> float:
    t = _finite_1d(tve_array)
    return float(np.max(t)) if len(t) > 0 else float("nan")

def tve_rmse(tve_array: Any) -> float:
    return rmse(tve_array)

def tve_outlier_rate(tve_array: Any, threshold_pct: float = 1.0) -> float:
    t = _finite_1d(tve_array)
    if len(t) == 0:
        return float("nan")
    return float(np.mean(t > threshold_pct))
