from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List, Sequence

# ============================================================
# 1. Configuración de Rigor (IEEE C37.118 / C37.242)
# ============================================================

@dataclass(frozen=True)
class MetricConfig:
    """Configuración de métricas para Journal Q1 y cumplimiento IEEE."""
    fs_hz: float
    f_nom: float = 60.0

    # Límites IEEE C37.118.1 para M-Class (Protección/Control)
    ieee_fe_limit_mhz: float = 5.0    # 5 mHz límite de error de frecuencia
    ieee_rfe_limit_hzs: float = 0.1   # 0.1 Hz/s límite de error de ROCOF
    
    # Umbrales industriales para análisis dinámico
    trip_thresholds_hz: Tuple[float, ...] = (0.2, 0.5)
    settling_tols_hz: Tuple[float, ...] = (0.02, 0.05)

    # RoCoF (Rate of Change of Frequency)
    rocof_smoothing_win: int = 50
    rocof_unit: str = "Hz/s"

    # Robustez Estadística
    percentiles: Tuple[float, ...] = (5, 50, 95, 99)
    cvar_levels: Tuple[float, ...] = (95, 99)

JSONValue = Union[None, bool, str, float, int, Dict[str, Any], List[Any]]

# ============================================================
# 2. Helpers de Serialización y Robustez Estadística
# ============================================================

def _mk_metric(name: str, value_raw: Any, scenario_id: str = "unknown", **kwargs) -> Dict[str, JSONValue]:
    """Crea un registro métrico rico compatible con el JSON del Runner."""
    raw = float(value_raw)
    val = float(raw) if math.isfinite(raw) else None

    status_code, reason = "SUCCESS", "Computed successfully."
    if math.isinf(raw):
        status_code = "DIVERGED"
        reason = "Numerical instability / Divergence detected."
    elif math.isnan(raw):
        status_code = "NOT_APPLICABLE"
        reason = "No event or invalid data for this scenario."

    return {
        "name": name, 
        "value": val, 
        "raw": raw,
        "status": {"code": status_code, "reason": reason, "triggered": val is not None},
        "spec": {
            "units": kwargs.get("units", ""),
            "threshold": kwargs.get("threshold"),
            "formula": kwargs.get("formula", ""),
        }
    }

def _percentile(x: np.ndarray, p: float) -> float:
    f = x[np.isfinite(x)]
    return float(np.percentile(f, p)) if f.size else float("nan")

def _cvar(x: np.ndarray, level: float) -> float:
    """Conditional Value at Risk (Promedio del peor caso/cola de error)."""
    f = x[np.isfinite(x)]
    if not f.size: return float("nan")
    q = np.percentile(f, level)
    tail = f[f >= q]
    return float(np.mean(tail)) if tail.size else float(q)

# ============================================================
# 3. Primitivas de Error (Core)
# ============================================================

def rmse(err: np.ndarray) -> float:
    e = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")

def mae(err: np.ndarray) -> float:
    e = err[np.isfinite(err)]
    return float(np.mean(np.abs(e))) if e.size else float("nan")

def fe_max_mhz(err: np.ndarray) -> float:
    """Frequency Error Máximo en mHz (Métrica IEEE)."""
    e = err[np.isfinite(err)]
    return float(np.max(np.abs(e)) * 1000.0) if e.size else float("nan")

def iae_abs(err: np.ndarray, fs_hz: float) -> float:
    """Integral of Absolute Error (Balance energético del error)."""
    e = err[np.isfinite(err)]
    return float(np.sum(np.abs(e)) / fs_hz) if e.size else float("nan")

# ============================================================
# 4. Dinámica y Transitorios (Foco Chamorro y Protección)
# ============================================================

def rocof_hz_per_s(f: np.ndarray, fs: float, smoothing_win: int = 1) -> np.ndarray:
    """Cálculo de ROCOF con suavizado para evitar ruido de derivada."""
    if smoothing_win > 1:
        k = np.ones(smoothing_win) / smoothing_win
        f = np.convolve(f, k, mode='same')
    rocof = np.concatenate(([0], np.diff(f) * fs))
    return rocof

def settling_time(err: np.ndarray, fs_hz: float, tol_hz: float) -> float:
    """Tiempo de asentamiento tras disturbio."""
    ae = np.abs(err)
    ae[~np.isfinite(ae)] = np.inf
    idx = np.where(ae > tol_hz)[0]
    last_violation = idx[-1] if idx.size > 0 else -1
    return float((last_violation + 1) / fs_hz)

def trip_time_threshold(f_est: np.ndarray, f_nom: float, fs_hz: float, thr: float) -> float:
    """Primer instante t donde la desviación de frecuencia cruza el umbral."""
    dev = np.abs(f_est - f_nom)
    # Evitar disparos por NaNs iniciales (warm-up)
    dev[~np.isfinite(dev)] = 0.0 
    idx = np.where(dev >= thr)[0]
    return float(idx[0] / fs_hz) if idx.size > 0 else float("inf")

def nadir_metrics(f_true: np.ndarray, f_est: np.ndarray, fs_hz: float) -> Dict[str, float]:
    """Calcula el error en el Nadir de frecuencia (Métrica Chamorro)."""
    idx_true = np.argmin(f_true)
    idx_est = np.argmin(f_est)
    
    val_err = float(f_est[idx_est] - f_true[idx_true])
    time_err_ms = float((idx_est - idx_true) / fs_hz * 1000.0)
    
    return {
        "nadir_val_err_hz": val_err,
        "nadir_time_err_ms": time_err_ms
    }

def overshoot_hz(f_true: np.ndarray, f_est: np.ndarray) -> float:
    """Máximo sobreimpulso detectado tras la recuperación de frecuencia."""
    e = f_est - f_true
    return float(np.max(e)) if e.size else float("nan")

# ============================================================
# 5. Agregación y Calidad Estadística
# ============================================================

@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    ieee_compliance: bool
    n_finite: int

def summarize(vals: Sequence[float] | np.ndarray, limit: float = 1e9) -> SummaryStats:
    v = np.asarray(vals)
    fin = v[np.isfinite(v)]
    if not fin.size: 
        return SummaryStats(0.0, 0.0, 0.0, 0.0, 0.0, False, 0)
    
    avg = float(np.mean(fin))
    return SummaryStats(
        mean=avg,
        std=float(np.std(fin)),
        p5=float(np.percentile(fin, 5)),
        p50=float(np.percentile(fin, 50)),
        p95=float(np.percentile(fin, 95)),
        ieee_compliance=bool(avg <= limit),
        n_finite=len(fin)
    )