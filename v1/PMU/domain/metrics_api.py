from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .metrics_base import (
    MetricConfig, 
    _mk_metric, 
    _percentile, 
    _cvar
)
from . import metrics_logic as logic

def _to_1d(x: Any) -> np.ndarray:
    """Conversión segura a array 1D de floats para evitar errores de dimensión."""
    return np.asarray(x, dtype=float).reshape(-1)

def compute_metrics(
    f_hat: np.ndarray, 
    f_true: np.ndarray, 
    exec_time_s: float, 
    latency_samples: int, 
    cfg: MetricConfig, 
    scenario_id: str = "unknown"
) -> Dict[str, Dict[str, Any]]:
    """
    API de alto nivel para el cálculo de métricas. 
    Organiza los datos y aplica el rigor de 'Warm-up' antes de llamar a la lógica.
    """
    fs = float(cfg.fs_hz)
    f_hat_full = _to_1d(f_hat)
    f_true_full = _to_1d(f_true)
    n_total = int(min(f_hat_full.size, f_true_full.size))
    
    # --- 1. Warm-up (Estabilización inicial) ---
    # Ignoramos los primeros 100ms para evitar que el transitorio de arranque 
    # del filtro (filtro de fase, etc.) contamine las métricas de estado estable.
    warm_up = int(0.1 * fs) 
    if n_total <= warm_up: warm_up = 0
    
    f_hat_clean = f_hat_full[warm_up:n_total]
    f_true_clean = f_true_full[warm_up:n_total]
    err_clean = f_hat_clean - f_true_clean
    abs_err_clean = np.abs(err_clean)
    
    out: Dict[str, Dict[str, Any]] = {}

    # Fail-safe para señales vacías o excesivamente cortas
    if len(f_hat_clean) <= 0:
        return {k: _mk_metric(k, float("nan"), scenario_id) for k in ["RMSE", "FE_max_mHz"]}

    # --- 2. Métricas Estándar y de Precisión (Sobre señal limpia) ---
    out["RMSE"] = _mk_metric("RMSE", logic.rmse(f_hat_clean, f_true_clean), scenario_id, units="Hz")
    out["MAE"] = _mk_metric("MAE", logic.mae(f_hat_clean, f_true_clean), scenario_id, units="Hz")
    
    # Métrica Clave IEEE C37.118.1 (Frequency Error en mHz)
    out["FE_max_mHz"] = _mk_metric("FE_max_mHz", logic.frequency_error_mhz(f_hat_clean, f_true_clean), 
                                   scenario_id, units="mHz", threshold=cfg.ieee_fe_limit_mhz)

    # --- 3. Métricas de ROCOF y Dinámica ---
    out["RFE_RMSE"] = _mk_metric("RFE_RMSE", logic.rocof_error_rmse(f_hat_clean, f_true_clean, fs), 
                                 scenario_id, units="Hz/s^2", threshold=cfg.ieee_rfe_limit_hzs)
    
    out["Settling_Time"] = _mk_metric("Settling_Time", 
                                      logic.settling_time(f_hat_clean, f_true_clean, cfg.settling_tols_hz[0], fs),
                                      scenario_id, units="s")

    # --- 4. Análisis de Nadir (Foco Chamorro Paper) ---
    # Captura la precisión en el punto más crítico de la caída de frecuencia tras la falla
    nadir = logic.nadir_analysis(f_hat_clean, f_true_clean, fs)
    out["Nadir_Mag_Err"] = _mk_metric("Nadir_Mag_Err", nadir["nadir_val_err_hz"], scenario_id, units="Hz")
    out["Nadir_Time_Err"] = _mk_metric("Nadir_Time_Err", nadir["nadir_time_err_ms"], scenario_id, units="ms")
    out["Overshoot"] = _mk_metric("Overshoot", logic.overshoot(f_hat_clean, f_true_clean), scenario_id, units="Hz")

    # --- 5. Tiempos de Disparo (Protección sobre señal COMPLETA) ---
    # Importante: Evaluamos sobre f_hat_full porque la latencia real de disparo 
    # se cuenta desde t=0, no desde el final del warm-up.
    for thr in cfg.trip_thresholds_hz:
        val, info = logic.trip_time(f_hat_full, f_true_full, thr, fs)
        key = f"TRIP_TIME_{str(thr).replace('.', 'p')}"
        out[key] = _mk_metric(key, val, scenario_id, units="s", threshold=thr)

    # --- 6. Robustez Estadística (Peor Caso) ---
    # Para Journal Q1: Reportamos percentiles de error absoluto para ver el "Tail Error"
    for p in cfg.percentiles:
        p_val = _percentile(abs_err_clean, p)
        out[f"P{int(p)}_ABS_ERR"] = _mk_metric(f"P{int(p)}_ABS_ERR", p_val, scenario_id, units="Hz")
    
    for lv in cfg.cvar_levels:
        cvar_val = _cvar(abs_err_clean, lv)
        out[f"CVAR{int(lv)}_ABS_ERR"] = _mk_metric(f"CVAR{int(lv)}_ABS_ERR", cvar_val, scenario_id, units="Hz")

    # --- 7. Costo Computacional ---
    tps_us = (float(exec_time_s) / n_total) * 1e6
    out["TIME_PER_SAMPLE_US"] = _mk_metric("TIME_PER_SAMPLE_US", tps_us, scenario_id, units="us")
    out["LATENCY_SAMPLES"] = _mk_metric("LATENCY_SAMPLES", float(latency_samples), scenario_id, units="samples")

    return out