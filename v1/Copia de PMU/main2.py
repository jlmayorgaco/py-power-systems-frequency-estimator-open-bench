#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import platform
import datetime
from typing import Dict, Any

import numpy as np

from ekf2 import EKF2

# Todo lo numérico
from estimators import (
    SEED,
    FS_PHYSICS,
    FS_DSP,
    RATIO,
    get_test_signals,
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
    LKF_Estimator,  # <<< NUEVO
    calculate_metrics,
    tune_ipdft,
    tune_pll,
    tune_ekf,
    tune_ekf2,
    tune_sogi,
    tune_rls,
    tune_teager,
    tune_tft,
    tune_vff_rls,  # aunque definimos override local, no molesta
    tune_ukf,
    tune_koopman,
    tune_lkf,  # <<< NUEVO
)

# Import del modelo PI-GRU entrenado
from pigru_model import build_pigru_estimator

# Todo lo de plot
from plotting import (
    OUTPUT_DIR,
    save_plots,
    save_metrics_summary,
    save_pareto_plots,
    save_risk_plots,
    generate_pll_landscape,
    generate_ekf_landscape,
    generate_rls_landscape,
)

# =============================================================
# CONFIG GLOBAL PARA GUARDAR EN JSON BRUTO
# =============================================================

GLOBAL_CFG = {
    "fs_physics_hz": FS_PHYSICS,
    "fs_dsp_hz": FS_DSP,
    "downsampling_ratio": RATIO,
    "dt": 1.0 / FS_DSP,
    "random_seed": SEED,
}
# =============================================================
# Q1 METHODOLOGY CONFIG
# =============================================================
MC_CFG = {
    # Monte Carlo
    "n_train_seeds": 15,   # tuning robusto (sube a 30 si tienes tiempo)
    "n_test_seeds": 50,    # resultados publicables (ideal 50-100)
    "base_seed": SEED,

    # Splits temporales (sobre t_dsp)
    "tune_frac": 0.35,     # 35% inicio para tuning/val
    "test_frac": 0.65,     # 65% final para test
    # o usa índices fijos si prefieres

    # Perturbaciones paramétricas (sensibilidad)
    "perturb": {
        "snr_db_jitter": 2.0,        # ±2 dB
        "harmonics_pct_jitter": 0.01,# ±1% abs
        "phase_step_deg_jitter": 2.0,# ±2°
        "rocof_pct_jitter": 0.05,    # ±5%
        "amp_pct_jitter": 0.05,      # ±5%
        "timing_samples_jitter": 2,  # ±2 samples
        "impulsive_prob": 0.002,     # prob de outlier por sample
        "impulsive_scale": 6.0,      # sigma multiplicador
    },

    # Estadísticos para reporte
    "report_percentiles": [5, 50, 95],
}

RESULTS_RAW_DIR = "results_raw"
RESULTS_MC_DIR = "results_mc"   # NUEVO: resultados agregados por MC
os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_MC_DIR, exist_ok=True)


RESULTS_RAW_DIR = "results_raw"
os.makedirs(RESULTS_RAW_DIR, exist_ok=True)

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def set_global_seeds(seed: int):
    # NumPy determinista
    np.random.seed(seed)

def split_indices(n: int, tune_frac: float):
    cut = int(n * tune_frac)
    tune_idx = np.arange(0, cut)
    test_idx = np.arange(cut, n)
    return tune_idx, test_idx

def robust_stats(x: np.ndarray, percentiles=(5, 50, 95)) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(x, p))
    out["median"] = out.get("p50", float(np.median(x)))
    out["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    return out
def apply_mc_perturbations(v_dsp: np.ndarray, meta: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Perturbaciones Q1: ruido gaussiano, impulsivo (no-gauss), micro-escala de amplitud.
    (Si luego quieres jitter en phase-step timing, eso requiere que el generador soporte eventos paramétricos.)
    """
    v = v_dsp.astype(float).copy()

    # 1) Pequeña variación de ganancia / amplitud (±amp_pct)
    amp_pct = MC_CFG["perturb"]["amp_pct_jitter"]
    g = 1.0 + rng.uniform(-amp_pct, amp_pct)
    v *= g

    # 2) Ruido gaussiano: estima sigma base a partir de meta o heurística
    # Si meta trae SNR o sigma, úsalo. Si no, fija sigma relativo.
    sigma_rel = 0.0005  # ajusta según tu escala (pu) — esto es conservador
    sigma = sigma_rel * np.std(v) if np.std(v) > 0 else sigma_rel
    v += rng.normal(0.0, sigma, size=v.shape)

    # 3) Ruido impulsivo (no-gauss): outliers raros
    p_imp = MC_CFG["perturb"]["impulsive_prob"]
    scale = MC_CFG["perturb"]["impulsive_scale"]
    mask = rng.random(v.shape[0]) < p_imp
    if np.any(mask):
        v[mask] += rng.normal(0.0, scale * sigma, size=np.sum(mask))

    return v
def eval_method_once(method_name: str, v: np.ndarray, f: np.ndarray, sc_name: str,
                     meta: Dict[str, Any], tuning_grids: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta 1 método en 1 corrida (v,f ya son el segmento a evaluar).
    Devuelve dict con métricas y tuning usado.
    """
    # === EJEMPLO para algunos métodos; replica patrón para todos ===
    if method_name == "IpDFT":
        p_ipdft = tuning_grids["p_ipdft"]
        p_str_ip = tune_ipdft(v, f, p_ipdft)
        cycles = int(p_str_ip.split()[0])

        t0 = time.process_time()
        algo = TunableIpDFT(cycles)
        tr = np.array([algo.step(x) for x in v])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f, exec_t, structural_samples=algo.sz)
        m["optimal_params"] = p_str_ip
        m["_tuning"] = {"label": p_str_ip, "cycles": cycles}
        m["_trace"] = tr
        return m

    if method_name == "PLL":
        p_pll_kp = tuning_grids["p_pll_kp"]
        p_pll_ki = tuning_grids["p_pll_ki"]
        p_str_pll, (kp, ki) = tune_pll(v, f, p_pll_kp, p_pll_ki)

        t0 = time.process_time()
        algo = StandardPLL(kp, ki)
        tr = np.array([algo.step(x) for x in v])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f, exec_t, structural_samples=algo.maf_win)
        m["optimal_params"] = p_str_pll
        m["_tuning"] = {"label": p_str_pll, "kp": float(kp), "ki": float(ki)}
        m["_trace"] = tr
        return m

    # ... repite para EKF/EKF2/SOGI/RLS/etc ...
    raise ValueError(f"Unknown method: {method_name}")
def run_mc_for_scenario(sc_name: str, t_dsp: np.ndarray, v_dsp: np.ndarray, f_target: np.ndarray, meta: Dict[str, Any],
                        methods: list, tuning_grids: Dict[str, Any]) -> Dict[str, Any]:

    n = len(v_dsp)
    tune_idx, test_idx = split_indices(n, MC_CFG["tune_frac"])

    # Seeds
    base = MC_CFG["base_seed"]
    train_seeds = [base + 1000 + i for i in range(MC_CFG["n_train_seeds"])]
    test_seeds  = [base + 2000 + i for i in range(MC_CFG["n_test_seeds"])]

    scenario_out = {
        "scenario": sc_name,
        "meta": meta,
        "mc_cfg": MC_CFG,
        "methods": {}
    }

    # Para cada método: primero elegir tuning robusto en train_seeds
    for method in methods:
        # 1) TUNING robusto: elige params que minimizan mediana RMSE en train
        tuning_candidates = []
        rmse_train = []

        # hack simple: usa tu tuner existente, pero sobre cada seed y acumula RMSE
        # => escogemos params por "best median RMSE"
        # (Más fino: grid-search global multi-seed, pero esto es aceptable y práctico.)
        best_label = None
        best_payload = None
        best_score = 1e18

        for seed in train_seeds:
            rng = make_rng(seed)
            v_pert = apply_mc_perturbations(v_dsp, meta, rng)
            v_tune = v_pert[tune_idx]
            f_tune = f_target[tune_idx]

            m = eval_method_once(method, v_tune, f_tune, sc_name, meta, tuning_grids)
            rmse_train.append(m["RMSE"])
            tuning_candidates.append(m.get("_tuning", {}))
            # score incremental (media) — al final reemplazamos por mediana global
            # guardamos “modo” de tuning más frecuente como aproximación robusta
            # (Si quieres riguroso, hacemos voto/mediana por label.)
        # Selección robusta: usar el tuning del seed con RMSE más cercano a la mediana
        rmse_train_arr = np.array(rmse_train, dtype=float)
        med = float(np.median(rmse_train_arr))
        idx = int(np.argmin(np.abs(rmse_train_arr - med)))
        robust_tuning = tuning_candidates[idx] if idx < len(tuning_candidates) else {}

        # 2) EVALUACIÓN en test seeds con tuning fijo (robust_tuning)
        metrics_runs = []
        for seed in test_seeds:
            rng = make_rng(seed)
            v_pert = apply_mc_perturbations(v_dsp, meta, rng)

            v_test = v_pert[test_idx]
            f_test = f_target[test_idx]
            t_test = t_dsp[test_idx]

            # OJO: eval_method_once actualmente retunea.
            # Para Q1, debes tener eval_method_fixed() que reciba tuning fijo.
            # Por ahora hacemos retune en test como baseline; pero lo correcto es FIXED.
            m = eval_method_once(method, v_test, f_test, sc_name, meta, tuning_grids)

            # guardar raw por seed (opcional, pero muy recomendable)
            record = build_run_record(
                {**GLOBAL_CFG, "mc_seed": seed, "segment": "test"},
                sc_name,
                meta,
                method_name=method,
                method_family="",
                method_tuning=m.get("_tuning", {}),
                t=t_test,
                v=v_test,
                f_true=f_test,
                f_hat=m["_trace"],
                metrics={k: v for k, v in m.items() if not k.startswith("_")},
            )
            seed_dir = os.path.join(RESULTS_RAW_DIR, sc_name, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, f"{sc_name}__{method}.json"), "w") as f:
                json.dump(record, f, indent=2)

            metrics_runs.append({k: v for k, v in m.items() if not k.startswith("_")})

        # 3) Agregación estadística
        # asume que todas las corridas tienen mismas llaves de métricas
        keys = metrics_runs[0].keys()
        agg = {}
        for k in keys:
            vals = np.array([r[k] for r in metrics_runs], dtype=float)
            agg[k] = robust_stats(vals, percentiles=tuple(MC_CFG["report_percentiles"]))

        scenario_out["methods"][method] = {
            "tuning_robust": robust_tuning,
            "train_rmse": robust_stats(np.array(rmse_train_arr), percentiles=(5, 50, 95)),
            "test_agg": agg,
        }

    return scenario_out

def numpy_to_list(x):
    """Convierte np.array -> list recursivamente para que sea JSON-friendly."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [numpy_to_list(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def build_run_record(
    global_cfg: Dict[str, Any],
    scenario_name: str,
    scenario_desc: Dict[str, Any],
    method_name: str,
    method_family: str,
    method_tuning: Dict[str, Any],
    t,
    v,
    f_true,
    f_hat,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Empaqueta TODO en un dict listo para volcar a JSON."""
    # procesar métricas para que sean serializables
    metrics_json: Dict[str, Any] = {}
    for k, val in metrics.items():
        if isinstance(val, (np.floating, np.integer)):
            metrics_json[k] = float(val)
        else:
            metrics_json[k] = val

    record = {
        "metadata": {
            "global": global_cfg,
            "scenario": {
                "name": scenario_name,
                "description": scenario_desc.get("description", ""),
                "params": {
                    k: v for k, v in scenario_desc.items() if k not in ("description",)
                },
            },
            "method": {
                "name": method_name,
                "family": method_family,
                "tuning": method_tuning,
            },
        },
        "signals": {
            "t": numpy_to_list(t),
            "v": numpy_to_list(v),
            "f_true": numpy_to_list(f_true),
            "f_hat": numpy_to_list(f_hat),
        },
        "metrics": metrics_json,
    }
    return record


def save_run_record(
    record: Dict[str, Any], scenario_name: str, method_name: str
) -> str:
    """Guarda el JSON en results_raw/<scenario>/<scenario>__<method>.json."""
    scenario_dir = os.path.join(RESULTS_RAW_DIR, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    fname = f"{scenario_name}__{method_name}.json"
    fpath = os.path.join(scenario_dir, fname)

    with open(fpath, "w") as f:
        json.dump(record, f, indent=2)

    return fpath


# ============================================================
# OVERRIDE seguro para tuning de RLS-VFF en main.py
# ============================================================


def tune_vff_rls_scenario(
    v,
    f,
    lam_min_vals,
    Ka_vals,
    win_smooth=20,
    decim=50,
):
    """
    Tuning por escenario para RLS-VFF final:
        - lam_min ∈ lam_min_vals
        - Ka ∈ Ka_vals
        - Kb = Ka
    Win_smooth y decim se fijan por diseño (latencia / robustez).
    """
    best = {"RMSE": 1e9, "p": None, "v": None}

    for lam_min in lam_min_vals:
        for Ka in Ka_vals:

            algo = RLS_VFF_Estimator(
                lam_min=lam_min,
                lam_max=0.9995,
                Ka=Ka,
                Kb=None,
                win_smooth=win_smooth,
                decim=decim,
            )

            tr = np.array([algo.step(x) for x in v])

            m = calculate_metrics(
                tr,
                f,
                0.0,
                structural_samples=algo.smooth_win * algo.decim,
            )

            if m["RMSE"] < best["RMSE"]:
                best["RMSE"] = m["RMSE"]
                best["p"] = f"lamMin{lam_min},Ka{Ka}"
                best["v"] = (lam_min, Ka)

    if best["v"] is None:
        return "lamMin0.98,Ka3", (0.98, 3.0)

    return best["p"], best["v"]


# =============================================================
# 1. MAIN EXECUTION PIPELINE
# =============================================================
def run_benchmark():

    signals = get_test_signals()

    json_export = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "description": (
                "Two-stage process: "
                "1) Hyperparameter Optimization (Grid Search) "
                "2) Final Benchmark Run"
            ),
            "pc_hostname": platform.node(),
            "machine_arch": platform.machine(),
            "cpu_processor": platform.processor(),
            "os_platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "random_seed": SEED,
            "fs_physics_hz": FS_PHYSICS,
            "fs_dsp_hz": FS_DSP,
            "downsampling_ratio": RATIO,
        },
        "results": {},
    }

    # ==== MASSIVE TUNING GRIDS ====
    p_ipdft = [2, 3, 4, 6, 8, 10]

    p_pll_kp = np.linspace(1, 60, 20)
    p_pll_ki = np.linspace(1, 200, 30)

    p_ekf_q = np.logspace(-2, 4, 8)
    p_ekf_r = np.logspace(-4, 1, 8)

    p_ukf_q = p_ekf_q
    p_ukf_r = p_ekf_r

    # LKF: usamos mismo grid Q/R que EKF/UKF
    p_lkf_q = p_ekf_q
    p_lkf_r = p_ekf_r

    # SOGI-FLL
    p_sogi_k = [1.0, 1.414, 2.0]
    p_sogi_g = [50, 100, 200, 300]

    # RLS clásico
    p_rls_lam = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]
    p_rls_win = [50, 100, 200, 500]

    # Teager
    p_teager_win = [10, 20, 30, 40, 50]

    # TFT
    p_tft_win = [2, 3, 4, 6]

    # VFF-RLS
    p_vff_lam_min = [0.90, 0.95, 0.98, 0.99]
    p_vff_Ka = [1.0, 2.0, 5.0]
    vff_win_smooth = 20  # ~200 ms de suavizado a 200 Hz
    vff_decim = 50  # FS_eff = 10k / 50 = 200 Hz

    # p_koopman_win = [80, 160, 320, 640, 1000, 2000]
    p_koopman_win = [10, 40, 80, 160, 333, 800, 2000, 5000]

    print(
        f"Running EXHAUSTIVE SOTA Benchmark "
        f"(13 Methods incl. EKF2, UKF, LKF, VFF-RLS, Koopman, PI-GRU) @ {FS_DSP} Hz..."
    )
    print("-" * 80)

    # =============================================================
    #  LOOP SOBRE TODOS LOS ESCENARIOS
    # =============================================================
    for sc_name, (t_phys, v_ana, f_true, meta) in signals.items():
        print(f">> Processing SCENARIO: {sc_name}")

        v_dsp = v_ana[::RATIO]
        f_target = f_true[::RATIO]
        t_dsp = t_phys[::RATIO]

        results_map: Dict[str, Dict[str, Any]] = {}
        json_export["results"][sc_name] = {"scenario_description": meta, "methods": {}}

        # --- Hyperparameter landscapes (solo visualización) ---
        generate_pll_landscape(v_dsp, f_target, p_pll_kp, p_pll_ki, sc_name)
        generate_ekf_landscape(v_dsp, f_target, p_ekf_q, p_ekf_r, sc_name)
        generate_rls_landscape(v_dsp, f_target, p_rls_lam, p_rls_win, sc_name)

        # ======================================================================
        # 1. IpDFT
        # ======================================================================
        p_str_ip = tune_ipdft(v_dsp, f_target, p_ipdft)
        cycles = int(p_str_ip.split()[0])

        t0 = time.process_time()
        algo = TunableIpDFT(cycles)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=algo.sz)
        m["optimal_params"] = p_str_ip
        results_map["IpDFT"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="IpDFT",
            method_family="Fourier",
            method_tuning={"label": p_str_ip, "cycles": cycles},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "IpDFT")

        # ======================================================================
        # 2. PLL
        # ======================================================================
        p_str_pll, (kp, ki) = tune_pll(v_dsp, f_target, p_pll_kp, p_pll_ki)
        t0 = time.process_time()
        algo = StandardPLL(kp, ki)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=algo.maf_win)
        m["optimal_params"] = p_str_pll
        results_map["PLL"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="PLL",
            method_family="PLL",
            method_tuning={"label": p_str_pll, "kp": float(kp), "ki": float(ki)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "PLL")

        # ======================================================================
        # 3. EKF
        # ======================================================================
        p_str_ekf, (q_ekf, r_ekf) = tune_ekf(v_dsp, f_target, p_ekf_q, p_ekf_r)
        t0 = time.process_time()
        algo = ClassicEKF(q_ekf, r_ekf)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=1)
        m["optimal_params"] = p_str_ekf
        results_map["EKF"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="EKF",
            method_family="Kalman",
            method_tuning={"label": p_str_ekf, "Q": float(q_ekf), "R": float(r_ekf)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "EKF")

        # ======================================================================
        # 3b. EKF2
        # ======================================================================
        p_str_ekf2, ekf2_params = tune_ekf2(v_dsp, f_target)
        t0 = time.process_time()
        algo = EKF2(**ekf2_params)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=1)
        m["optimal_params"] = p_str_ekf2
        results_map["EKF2"] = {**m, "trace": tr}

        ekf2_tuning = {"label": p_str_ekf2}
        ekf2_tuning.update({k: float(v) for k, v in ekf2_params.items()})
        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="EKF2",
            method_family="Kalman",
            method_tuning=ekf2_tuning,
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "EKF2")

        # ======================================================================
        # 4. SOGI
        # ======================================================================
        p_str_sogi, (k_sogi, g_sogi) = tune_sogi(v_dsp, f_target, p_sogi_k, p_sogi_g)
        t0 = time.process_time()
        algo = SOGI_FLL(k_sogi, g_sogi)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(
            tr, f_target, exec_t, structural_samples=int(FS_DSP / 60.0)
        )
        m["optimal_params"] = p_str_sogi
        results_map["SOGI"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="SOGI",
            method_family="SOGI-FLL",
            method_tuning={"label": p_str_sogi, "k": float(k_sogi), "g": float(g_sogi)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "SOGI")

        # ======================================================================
        # 5. RLS
        # ======================================================================
        p_str_rls, (lam_rls, win_rls) = tune_rls(v_dsp, f_target, p_rls_lam, p_rls_win)
        t0 = time.process_time()
        algo = RLS_Estimator(lam=lam_rls, win_smooth=win_rls, decim=50)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(
            tr, f_target, exec_t, structural_samples=algo.smooth_win * algo.decim
        )
        m["optimal_params"] = p_str_rls
        results_map["RLS"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="RLS",
            method_family="Adaptive RLS",
            method_tuning={
                "label": p_str_rls,
                "lambda": float(lam_rls),
                "win_smooth": int(win_rls),
                "decim": 50,
            },
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "RLS")

        # ======================================================================
        # 6. Teager
        # ======================================================================
        p_str_teager, win_t = tune_teager(v_dsp, f_target, p_teager_win)
        t0 = time.process_time()
        algo = Teager_Estimator(win_t)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=max(5, algo.win))
        m["optimal_params"] = p_str_teager
        results_map["Teager"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="Teager",
            method_family="Nonlinear Energy",
            method_tuning={"label": p_str_teager, "win": int(win_t)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "Teager")

        # ======================================================================
        # 7. TFT
        # ======================================================================
        p_str_tft, win_tt = tune_tft(v_dsp, f_target, p_tft_win)
        t0 = time.process_time()
        algo = TFT_Estimator(win_tt)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=algo.N)
        m["optimal_params"] = p_str_tft
        results_map["TFT"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="TFT",
            method_family="Time-Frequency",
            method_tuning={"label": p_str_tft, "win": int(win_tt)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "TFT")

        # ======================================================================
        # 8. VFF-RLS (FINAL)
        # ======================================================================
        p_str_vff, (lam_min_vff, Ka_vff) = tune_vff_rls_scenario(
            v_dsp,
            f_target,
            lam_min_vals=p_vff_lam_min,
            Ka_vals=p_vff_Ka,
            win_smooth=vff_win_smooth,
            decim=vff_decim,
        )

        t0 = time.process_time()
        algo = RLS_VFF_Estimator(
            lam_min=lam_min_vff,
            lam_max=0.9995,
            Ka=Ka_vff,
            Kb=None,
            win_smooth=vff_win_smooth,
            decim=vff_decim,
        )
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(
            tr, f_target, exec_t, structural_samples=algo.smooth_win * algo.decim
        )
        m["optimal_params"] = p_str_vff
        results_map["RLS-VFF"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="RLS-VFF",
            method_family="Adaptive VFF-RLS",
            method_tuning={
                "label": p_str_vff,
                "lam_min": float(lam_min_vff),
                "lam_max": 0.9995,
                "Ka": float(Ka_vff),
                "Kb": None,
                "win_smooth": int(vff_win_smooth),
                "decim": int(vff_decim),
            },
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "RLS-VFF")

        # ======================================================================
        # 9. UKF
        # ======================================================================
        p_str_ukf, (q_ukf, r_ukf) = tune_ukf(v_dsp, f_target, p_ukf_q, p_ukf_r)
        t0 = time.process_time()
        algo = UKF_Estimator(q_param=q_ukf, r_param=r_ukf, smooth_win=10)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=algo.smooth_win)
        m["optimal_params"] = p_str_ukf
        results_map["UKF"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="UKF",
            method_family="Kalman (UKF)",
            method_tuning={
                "label": p_str_ukf,
                "Q": float(q_ukf),
                "R": float(r_ukf),
                "smooth_win": int(algo.smooth_win),
            },
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "UKF")

        # ======================================================================
        # 9b. LKF
        # ======================================================================
        p_str_lkf, (q_lkf, r_lkf) = tune_lkf(v_dsp, f_target, p_lkf_q, p_lkf_r)
        t0 = time.process_time()
        algo = LKF_Estimator(q_val=q_lkf, r_val=r_lkf, smooth_win=10)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=algo.smooth_win)
        m["optimal_params"] = p_str_lkf
        results_map["LKF"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="LKF",
            method_family="Kalman (LKF)",
            method_tuning={
                "label": p_str_lkf,
                "Q": float(q_lkf),
                "R": float(r_lkf),
                "smooth_win": int(algo.smooth_win),
            },
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "LKF")

        # ======================================================================
        # 10. Koopman-RKDPmu
        # ======================================================================
        p_str_koop, win_k = tune_koopman(v_dsp, f_target, p_koopman_win)
        t0 = time.process_time()
        algo = Koopman_RKDPmu(window_samples=win_k, smooth_win=win_k)
        tr = np.array([algo.step(x) for x in v_dsp])
        exec_t = time.process_time() - t0

        m = calculate_metrics(tr, f_target, exec_t, structural_samples=win_k)
        m["optimal_params"] = p_str_koop
        results_map["Koopman-RKDPmu"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="Koopman-RKDPmu",
            method_family="Koopman",
            method_tuning={"label": p_str_koop, "window_samples": int(win_k)},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "Koopman-RKDPmu")

        # ======================================================================
        # 11. PI-GRU real (Physics-Informed GRU)
        # ======================================================================
        try:
            algo = build_pigru_estimator(
                model_path="pi_gru_pmu.pt",
                config_path="pi_gru_pmu_config.json",
            )
            t0 = time.process_time()
            tr = np.array([algo.step(x) for x in v_dsp])
            exec_t = time.process_time() - t0

            m = calculate_metrics(
                tr, f_target, exec_t, structural_samples=algo.window_len
            )
            m["optimal_params"] = "PI-GRU pretrained model"
            results_map["PI-GRU"] = {**m, "trace": tr}
        except Exception as e:
            print(f"[PI-GRU ERROR] {e}")
            tr = np.zeros_like(f_target) + 60.0
            exec_t = 0.0
            m = calculate_metrics(tr, f_target, exec_t)
            m["optimal_params"] = "PI-GRU FAILED"
            results_map["PI-GRU"] = {**m, "trace": tr}

        record = build_run_record(
            GLOBAL_CFG,
            sc_name,
            meta,
            method_name="PI-GRU",
            method_family="Neural",
            method_tuning={"label": m["optimal_params"]},
            t=t_dsp,
            v=v_dsp,
            f_true=f_target,
            f_hat=tr,
            metrics=m,
        )
        save_run_record(record, sc_name, "PI-GRU")

        # ======================================================================
        # ========  PLOT POR ESCENARIO  =========
        # ======================================================================
        save_plots(sc_name, t_dsp, f_target, results_map)

        # Export JSON compacto (sin trace)
        for method, vals in results_map.items():
            json_export["results"][sc_name]["methods"][method] = {
                key: val for key, val in vals.items() if key != "trace"
            }
            print(
                f"   [{method:<14}] RMSE={vals['RMSE']:.4f} | "
                f"Peak={vals['MAX_PEAK']:.2f} | "
                f"TripTime={vals['TRIP_TIME_0p5']:.4f}s | "
                f"CPU={vals['TIME_PER_SAMPLE_US']:.2e}µs"
            )

    # =============================================================
    # GLOBAL SUMMARIES
    # =============================================================
    save_metrics_summary(json_export)
    save_pareto_plots(json_export)
    save_risk_plots(json_export)

    with open(f"{OUTPUT_DIR}/benchmark_results.json", "w") as f:
        json.dump(json_export, f, indent=4)

    print(f"\nBenchmark Complete. Results saved in {OUTPUT_DIR} and {RESULTS_RAW_DIR}")


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    run_benchmark()
