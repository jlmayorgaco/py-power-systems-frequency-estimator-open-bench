from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, List
from .metrics_base import (
    MetricConfig, JSONValue, summarize,
    rmse as _rmse, mae as _mae, iae_abs as _iae_abs,
    fe_max_mhz as _fe_max_mhz, rocof_hz_per_s as _rocof_hz_per_s,
    settling_time as _settling_time, nadir_metrics as _nadir_metrics,
    overshoot_hz as _overshoot_hz, trip_time_threshold as _trip_time_threshold
)

# ============================================================
# 1. Primitivas de Error (Capa de Lógica)
# ============================================================

def rmse(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    return _rmse(f_hat - f_true)

def mae(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    return _mae(f_hat - f_true)

def frequency_error_mhz(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """Error de frecuencia máximo en mHz según estándar IEEE C37.118.1."""
    return _fe_max_mhz(f_hat - f_true)

def iae(f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float) -> float:
    """Integral del error absoluto (Integral of Absolute Error)."""
    return _iae_abs(f_hat - f_true, fs_hz)

def rocof_error_rmse(f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float) -> float:
    """
    RMSE del error de ROCOF (RFE). 
    Aplica un suavizado de 10 muestras para evitar el ruido de derivación pura.
    """
    r_true = _rocof_hz_per_s(f_true, fs_hz, smoothing_win=10)
    r_hat = _rocof_hz_per_s(f_hat, fs_hz, smoothing_win=10)
    return _rmse(r_hat - r_true)

# ============================================================
# 2. Análisis Dinámico y Nadir (Foco Chamorro y Protección)
# ============================================================

def nadir_analysis(f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float) -> Dict[str, float]:
    """Análisis detallado de precisión y latencia en el punto de frecuencia mínima."""
    return _nadir_metrics(f_true, f_hat, fs_hz)

def overshoot(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """Máximo error de sobreimpulso tras la recuperación del evento."""
    return _overshoot_hz(f_true, f_hat)

def settling_time(f_hat: np.ndarray, f_true: np.ndarray, tol_hz: float, fs_hz: float) -> float:
    """Tiempo que tarda el error en entrar y permanecer dentro de una banda de tolerancia."""
    err = f_hat - f_true
    return _settling_time(err, fs_hz, tol_hz)

def trip_time(f_hat: np.ndarray, f_true: np.ndarray, thr_hz: float, fs_hz: float) -> tuple[float, dict]:
    """
    Calcula el tiempo de disparo (Trip Time).
    Retorna el tiempo en segundos y un diccionario con metadatos de la detección.
    """
    # Se asume la frecuencia nominal como el primer valor estable de la referencia
    f_nom = float(f_true[0]) if f_true.size > 0 else 60.0
    val = _trip_time_threshold(f_hat, f_nom, fs_hz, thr_hz)
    
    status = "triggered" if np.isfinite(val) else "never_tripped"
    return float(val), {"reason": status, "f_nom": f_nom, "threshold": thr_hz}

# ============================================================
# 3. Agregación Monte-Carlo con Rigor Estadístico
# ============================================================

def aggregate_monte_carlo(
    per_seed_metrics: List[Dict[str, Dict[str, Any]]],
    config: MetricConfig
) -> Dict[str, Dict[str, JSONValue]]:
    """
    Agrega resultados de N semillas calculando estadísticas y cumplimiento de límites IEEE.
    """
    if not per_seed_metrics: return {}
    
    # Obtener todas las métricas calculadas en las semillas
    metric_names = set()
    for d in per_seed_metrics:
        metric_names |= set(d.keys())

    agg: Dict[str, Dict[str, JSONValue]] = {}

    for name in sorted(metric_names):
        vals_raw = []
        for d in per_seed_metrics:
            m = d.get(name)
            if m:
                # Extraemos el valor numérico para la agregación estadística
                # Intentamos obtener 'raw' para máxima precisión, si no 'value'
                val = m.get("raw") if m.get("raw") is not None else m.get("value")
                if val is not None:
                    vals_raw.append(float(val))
            else:
                vals_raw.append(float("nan"))

        # Definir límites de cumplimiento según la métrica para el reporte Q1
        limit = 1e9
        if "FE_max" in name: limit = config.ieee_fe_limit_mhz
        if "RFE" in name: limit = config.ieee_rfe_limit_hzs

        # Generar resumen estadístico con check de cumplimiento IEEE
        s = summarize(vals_raw, limit=limit)
        
        agg[name] = {
            "mean": s.mean,
            "std": s.std,
            "p5": s.p5,
            "p50": s.p50,
            "p95": s.p95,
            "ieee_compliance": s.ieee_compliance,
            "n_finite": s.n_finite
        }

    return agg

__all__ = [
    "rmse", "mae", "iae", "frequency_error_mhz", "rocof_error_rmse",
    "nadir_analysis", "overshoot", "settling_time", "trip_time", 
    "aggregate_monte_carlo"
]