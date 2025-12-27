# experiments/legacy_adapters.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Legacy imports (your current code)
from ekf2 import EKF2
from estimators import (
    TunableIpDFT,
    StandardPLL,
    ClassicEKF,
    SOGI_FLL,
    RLS_Estimator,
    Teager_Estimator,
    TFT_Estimator,
    RLS_VFF_Estimator,
    UKF_Estimator,
    Koopman_RKDPmu,
    LKF_Estimator,
)
from pigru_model import build_pigru_estimator


def factory_ipdft(tuning: Dict[str, Any]) -> Any:
    return TunableIpDFT(int(tuning["cycles"]))

def factory_pll(tuning: Dict[str, Any]) -> Any:
    return StandardPLL(float(tuning["kp"]), float(tuning["ki"]))

def factory_ekf(tuning: Dict[str, Any]) -> Any:
    return ClassicEKF(float(tuning["Q"]), float(tuning["R"]))

def factory_ekf2(tuning: Dict[str, Any]) -> Any:
    # EKF2 expects kwargs (your ekf2_params)
    params = dict(tuning)
    # keep label out if present
    params.pop("label", None)
    return EKF2(**params)

def factory_sogi(tuning: Dict[str, Any]) -> Any:
    return SOGI_FLL(float(tuning["k"]), float(tuning["g"]))

def factory_rls(tuning: Dict[str, Any]) -> Any:
    return RLS_Estimator(lam=float(tuning["lambda"]), win_smooth=int(tuning["win_smooth"]), decim=int(tuning["decim"]))

def factory_teager(tuning: Dict[str, Any]) -> Any:
    return Teager_Estimator(int(tuning["win"]))

def factory_tft(tuning: Dict[str, Any]) -> Any:
    return TFT_Estimator(int(tuning["win"]))

def factory_vff_rls(tuning: Dict[str, Any]) -> Any:
    return RLS_VFF_Estimator(
        lam_min=float(tuning["lam_min"]),
        lam_max=float(tuning.get("lam_max", 0.9995)),
        Ka=float(tuning["Ka"]),
        Kb=tuning.get("Kb", None),
        win_smooth=int(tuning["win_smooth"]),
        decim=int(tuning["decim"]),
    )

def factory_ukf(tuning: Dict[str, Any]) -> Any:
    return UKF_Estimator(
        q_param=float(tuning["Q"]),
        r_param=float(tuning["R"]),
        smooth_win=int(tuning.get("smooth_win", 10)),
    )

def factory_lkf(tuning: Dict[str, Any]) -> Any:
    return LKF_Estimator(
        q_val=float(tuning["Q"]),
        r_val=float(tuning["R"]),
        smooth_win=int(tuning.get("smooth_win", 10)),
    )

def factory_koopman(tuning: Dict[str, Any]) -> Any:
    w = int(tuning["window_samples"])
    return Koopman_RKDPmu(window_samples=w, smooth_win=w)

def factory_pigru(_: Dict[str, Any]) -> Any:
    # model config in files, not tuned here (you can later tune window_len etc.)
    return build_pigru_estimator(model_path="pi_gru_pmu.pt", config_path="pi_gru_pmu_config.json")


def latency_ipdft(tuning: Dict[str, Any]) -> int:
    # uses internal buffer size
    cycles = int(tuning["cycles"])
    # FS_DSP unknown here; we estimate later by reading algo.sz after init
    return 0

def latency_pll(tuning: Dict[str, Any]) -> int:
    # MAF 1-cycle is the main structural latency (algo.maf_win)
    return 0
