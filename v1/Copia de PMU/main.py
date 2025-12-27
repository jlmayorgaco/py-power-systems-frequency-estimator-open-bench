#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import platform
import datetime
from typing import Dict, Any, Tuple, List

import numpy as np

from ekf2 import EKF2

# =============================================================
# Estimators + tuners
# =============================================================
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
    LKF_Estimator,
    calculate_metrics,
    tune_ipdft,
    tune_pll,
    tune_ekf,
    tune_ekf2,
    tune_sogi,
    tune_rls,
    tune_teager,
    tune_tft,
    tune_ukf,
    tune_koopman,
    tune_lkf,
)

# PI-GRU
from pigru_model import build_pigru_estimator

# Plotting (TU pipeline existente)
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
# GLOBAL CONFIG
# =============================================================
GLOBAL_CFG = {
    "fs_physics_hz": FS_PHYSICS,
    "fs_dsp_hz": FS_DSP,
    "downsampling_ratio": RATIO,
    "dt": 1.0 / FS_DSP,
    "random_seed": SEED,
}

# =============================================================
# Q1 MONTE CARLO CONFIG
# =============================================================
MC_CFG = {
    "n_train_seeds": 15,
    "n_test_seeds": 50,
    "base_seed": SEED,

    "tune_frac": 0.35,

    "perturb": {
        "amp_pct_jitter": 0.05,
        "sigma_rel": 0.0005,
        "impulsive_prob": 0.002,
        "impulsive_scale": 6.0,
    },

    "report_percentiles": [5, 50, 95],
}

# =============================================================
# OUTPUT DIRS
# =============================================================
RESULTS_RAW_DIR = "results_raw"
RESULTS_MC_DIR = "results_mc"
os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_MC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# JSON helpers
# =============================================================
def numpy_to_list(x):
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
    metrics_json: Dict[str, Any] = {}
    for k, val in metrics.items():
        if isinstance(val, (np.floating, np.integer)):
            metrics_json[k] = float(val)
        else:
            metrics_json[k] = val

    return {
        "metadata": {
            "global": global_cfg,
            "scenario": {
                "name": scenario_name,
                "description": scenario_desc.get("description", ""),
                "params": {k: v for k, v in scenario_desc.items() if k != "description"},
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


# =============================================================
# MC utilities
# =============================================================
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def split_indices(n: int, tune_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    cut = int(n * tune_frac)
    return np.arange(0, cut), np.arange(cut, n)


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


def apply_mc_perturbations(v_dsp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = v_dsp.astype(float).copy()

    amp_pct = MC_CFG["perturb"]["amp_pct_jitter"]
    g = 1.0 + rng.uniform(-amp_pct, amp_pct)
    v *= g

    sigma_rel = MC_CFG["perturb"]["sigma_rel"]
    sigma = sigma_rel * np.std(v) if np.std(v) > 0 else sigma_rel
    v += rng.normal(0.0, sigma, size=v.shape)

    p_imp = MC_CFG["perturb"]["impulsive_prob"]
    scale = MC_CFG["perturb"]["impulsive_scale"]
    mask = rng.random(v.shape[0]) < p_imp
    if np.any(mask):
        v[mask] += rng.normal(0.0, scale * sigma, size=int(np.sum(mask)))

    return v


# =============================================================
# VFF-RLS tuning override (tu versión original)
# =============================================================
def tune_vff_rls_scenario(
    v,
    f,
    lam_min_vals,
    Ka_vals,
    win_smooth=20,
    decim=50,
):
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
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.smooth_win * algo.decim)
            if m["RMSE"] < best["RMSE"]:
                best["RMSE"] = m["RMSE"]
                best["p"] = f"lamMin{lam_min},Ka{Ka}"
                best["v"] = (lam_min, Ka)

    if best["v"] is None:
        return "lamMin0.98,Ka3", (0.98, 3.0)
    return best["p"], best["v"]


# =============================================================
# Tuning (train) + fixed run (test)
# =============================================================
def tune_method_on_segment(method: str, v_seg: np.ndarray, f_seg: np.ndarray, grids: Dict[str, Any]) -> Dict[str, Any]:
    if method == "IpDFT":
        p = tune_ipdft(v_seg, f_seg, grids["p_ipdft"])
        cycles = int(p.split()[0])
        return {"label": p, "cycles": cycles}

    if method == "PLL":
        p, (kp, ki) = tune_pll(v_seg, f_seg, grids["p_pll_kp"], grids["p_pll_ki"])
        return {"label": p, "kp": float(kp), "ki": float(ki)}

    if method == "EKF":
        p, (q, r) = tune_ekf(v_seg, f_seg, grids["p_ekf_q"], grids["p_ekf_r"])
        return {"label": p, "Q": float(q), "R": float(r)}

    if method == "EKF2":
        p, params = tune_ekf2(v_seg, f_seg)
        out = {"label": p}
        out.update({k: float(v) for k, v in params.items()})
        return out

    if method == "SOGI":
        p, (k_sogi, g_sogi) = tune_sogi(v_seg, f_seg, grids["p_sogi_k"], grids["p_sogi_g"])
        return {"label": p, "k": float(k_sogi), "g": float(g_sogi)}

    if method == "RLS":
        p, (lam, win) = tune_rls(v_seg, f_seg, grids["p_rls_lam"], grids["p_rls_win"])
        return {"label": p, "lambda": float(lam), "win_smooth": int(win), "decim": int(grids["rls_decim"])}

    if method == "Teager":
        p, win = tune_teager(v_seg, f_seg, grids["p_teager_win"])
        return {"label": p, "win": int(win)}

    if method == "TFT":
        p, win = tune_tft(v_seg, f_seg, grids["p_tft_win"])
        return {"label": p, "win": int(win)}

    if method == "RLS-VFF":
        p, (lam_min, Ka) = tune_vff_rls_scenario(
            v_seg,
            f_seg,
            lam_min_vals=grids["p_vff_lam_min"],
            Ka_vals=grids["p_vff_Ka"],
            win_smooth=grids["vff_win_smooth"],
            decim=grids["vff_decim"],
        )
        return {
            "label": p,
            "lam_min": float(lam_min),
            "lam_max": float(grids["vff_lam_max"]),
            "Ka": float(Ka),
            "Kb": None,
            "win_smooth": int(grids["vff_win_smooth"]),
            "decim": int(grids["vff_decim"]),
        }

    if method == "UKF":
        p, (q, r) = tune_ukf(v_seg, f_seg, grids["p_ukf_q"], grids["p_ukf_r"])
        return {"label": p, "Q": float(q), "R": float(r), "smooth_win": int(grids["kalman_smooth_win"])}

    if method == "LKF":
        p, (q, r) = tune_lkf(v_seg, f_seg, grids["p_lkf_q"], grids["p_lkf_r"])
        return {"label": p, "Q": float(q), "R": float(r), "smooth_win": int(grids["kalman_smooth_win"])}

    if method == "Koopman-RKDPmu":
        p, win_k = tune_koopman(v_seg, f_seg, grids["p_koopman_win"])
        return {"label": p, "window_samples": int(win_k), "smooth_win": int(win_k)}

    if method == "PI-GRU":
        return {"label": "PI-GRU pretrained model"}

    raise ValueError(f"Unknown method: {method}")


def run_method_fixed(method: str, v_seg: np.ndarray, tuning: Dict[str, Any]) -> Tuple[np.ndarray, float, int]:
    """
    Returns: (trace, exec_time_sec, structural_samples)
    """
    t0 = time.process_time()

    if method == "IpDFT":
        algo = TunableIpDFT(int(tuning["cycles"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.sz)

    if method == "PLL":
        algo = StandardPLL(float(tuning["kp"]), float(tuning["ki"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.maf_win)

    if method == "EKF":
        algo = ClassicEKF(float(tuning["Q"]), float(tuning["R"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, 1

    if method == "EKF2":
        kwargs = {k: v for k, v in tuning.items() if k != "label"}
        algo = EKF2(**kwargs)
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, 1

    if method == "SOGI":
        algo = SOGI_FLL(float(tuning["k"]), float(tuning["g"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(FS_DSP / 60.0)

    if method == "RLS":
        algo = RLS_Estimator(lam=float(tuning["lambda"]), win_smooth=int(tuning["win_smooth"]), decim=int(tuning["decim"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.smooth_win * algo.decim)

    if method == "Teager":
        algo = Teager_Estimator(int(tuning["win"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(max(5, algo.win))

    if method == "TFT":
        algo = TFT_Estimator(int(tuning["win"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.N)

    if method == "RLS-VFF":
        algo = RLS_VFF_Estimator(
            lam_min=float(tuning["lam_min"]),
            lam_max=float(tuning["lam_max"]),
            Ka=float(tuning["Ka"]),
            Kb=None,
            win_smooth=int(tuning["win_smooth"]),
            decim=int(tuning["decim"]),
        )
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.smooth_win * algo.decim)

    if method == "UKF":
        algo = UKF_Estimator(q_param=float(tuning["Q"]), r_param=float(tuning["R"]), smooth_win=int(tuning["smooth_win"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.smooth_win)

    if method == "LKF":
        algo = LKF_Estimator(q_val=float(tuning["Q"]), r_val=float(tuning["R"]), smooth_win=int(tuning["smooth_win"]))
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, int(algo.smooth_win)

    if method == "Koopman-RKDPmu":
        win = int(tuning["window_samples"])
        algo = Koopman_RKDPmu(window_samples=win, smooth_win=win)
        tr = np.array([algo.step(x) for x in v_seg])
        return tr, time.process_time() - t0, win

    if method == "PI-GRU":
        algo = build_pigru_estimator(model_path="pi_gru_pmu.pt", config_path="pi_gru_pmu_config.json")
        tr = np.array([algo.step(x) for x in v_seg])
        exec_t = time.process_time() - t0
        structural = int(getattr(algo, "window_len", getattr(algo, "window", 1)))
        return tr, exec_t, structural

    raise ValueError(f"Unknown method: {method}")


# =============================================================
# MC per scenario + representative run
# =============================================================
def run_mc_for_scenario(
    sc_name: str,
    t_dsp: np.ndarray,
    v_dsp: np.ndarray,
    f_target: np.ndarray,
    meta: Dict[str, Any],
    methods: List[str],
    grids: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - scenario_mc_summary
      - results_map_rep (para save_plots) con trace por método (representative seed)
    """
    n = len(v_dsp)
    tune_idx, test_idx = split_indices(n, MC_CFG["tune_frac"])

    base = MC_CFG["base_seed"]
    train_seeds = [base + 1000 + i for i in range(MC_CFG["n_train_seeds"])]
    test_seeds = [base + 2000 + i for i in range(MC_CFG["n_test_seeds"])]

    scenario_out: Dict[str, Any] = {
        "scenario": sc_name,
        "meta": meta,
        "mc_cfg": MC_CFG,
        "methods": {},
    }

    # representative traces for plotting (one seed)
    results_map_rep: Dict[str, Dict[str, Any]] = {}

    # For representative seed selection (per method)
    rep_seed_by_method: Dict[str, int] = {}

    for method in methods:
        # -------------------------
        # Train: tune per seed
        # -------------------------
        train_rmse = []
        train_tunings = []

        for seed in train_seeds:
            rng = make_rng(seed)
            v_pert = apply_mc_perturbations(v_dsp, rng)
            v_tune = v_pert[tune_idx]
            f_tune = f_target[tune_idx]

            tuning = tune_method_on_segment(method, v_tune, f_tune, grids)
            tr, exec_t, structural = run_method_fixed(method, v_tune, tuning)
            m = calculate_metrics(tr, f_tune, exec_t, structural_samples=structural)

            train_rmse.append(float(m["RMSE"]))
            train_tunings.append(tuning)

        train_rmse_arr = np.array(train_rmse, dtype=float)
        med_train = float(np.median(train_rmse_arr))
        idx = int(np.argmin(np.abs(train_rmse_arr - med_train)))
        robust_tuning = train_tunings[idx] if idx < len(train_tunings) else train_tunings[0]

        # -------------------------
        # Test: fixed tuning MC
        # -------------------------
        metrics_runs: List[Dict[str, Any]] = []
        rmse_test_list = []
        traces_cache: Dict[int, np.ndarray] = {}

        for seed in test_seeds:
            rng = make_rng(seed)
            v_pert = apply_mc_perturbations(v_dsp, rng)

            v_test = v_pert[test_idx]
            f_test = f_target[test_idx]
            t_test = t_dsp[test_idx]

            tr, exec_t, structural = run_method_fixed(method, v_test, robust_tuning)
            m = calculate_metrics(tr, f_test, exec_t, structural_samples=structural)

            metrics_runs.append(m)
            rmse_test_list.append(float(m["RMSE"]))
            traces_cache[seed] = tr

            # raw save per seed
            record = build_run_record(
                {**GLOBAL_CFG, "mc_seed": seed, "segment": "test"},
                sc_name,
                meta,
                method_name=method,
                method_family="",
                method_tuning=robust_tuning,
                t=t_test,
                v=v_test,
                f_true=f_test,
                f_hat=tr,
                metrics=m,
            )
            seed_dir = os.path.join(RESULTS_RAW_DIR, sc_name, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, f"{sc_name}__{method}.json"), "w") as f:
                json.dump(record, f, indent=2)

        rmse_test_arr = np.array(rmse_test_list, dtype=float)
        med_test = float(np.median(rmse_test_arr))
        rep_idx = int(np.argmin(np.abs(rmse_test_arr - med_test)))
        rep_seed = test_seeds[rep_idx]
        rep_seed_by_method[method] = rep_seed

        # aggregate stats
        keys = metrics_runs[0].keys()
        agg: Dict[str, Any] = {}
        for k in keys:
            vals = np.array([r[k] for r in metrics_runs], dtype=float)
            agg[k] = robust_stats(vals, percentiles=tuple(MC_CFG["report_percentiles"]))

        scenario_out["methods"][method] = {
            "tuning_robust": robust_tuning,
            "train_rmse": robust_stats(train_rmse_arr, percentiles=(5, 50, 95)),
            "test_agg": agg,
            "representative_seed": int(rep_seed),
        }

        # representative trace for plotting
        tr_rep = traces_cache[rep_seed]
        # rebuild metrics for rep trace on base f_target segment
        # NOTE: rep uses perturbed v, but for plotting we show against f_target[test]
        f_test = f_target[test_idx]
        t_test = t_dsp[test_idx]
        m_rep = calculate_metrics(tr_rep, f_test, 0.0, structural_samples=1)

        results_map_rep[method] = {**m_rep, "trace": tr_rep}

    # We’ll plot only test segment time axis
    scenario_out["_plot_segment"] = "test"
    scenario_out["_plot_indices"] = {"tune": [int(tune_idx[0]), int(tune_idx[-1])], "test": [int(test_idx[0]), int(test_idx[-1])]}

    return scenario_out, results_map_rep


# =============================================================
# Main benchmark
# =============================================================
def run_benchmark():
    signals = get_test_signals()

    json_export = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "description": (
                "Q1 methodology: Monte Carlo with temporal split. "
                "Robust tuning on train seeds; fixed evaluation on test seeds. "
                "Representative seed selected near median test RMSE for per-scenario plots."
            ),
            "pc_hostname": platform.node(),
            "machine_arch": platform.machine(),
            "cpu_processor": platform.processor(),
            "os_platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "global_cfg": GLOBAL_CFG,
            "mc_cfg": MC_CFG,
        },
        "results": {},
    }

    # ===== grids =====
    p_ipdft = [2, 3, 4, 6, 8, 10]
    p_pll_kp = np.linspace(1, 60, 20)
    p_pll_ki = np.linspace(1, 200, 30)

    p_ekf_q = np.logspace(-2, 4, 8)
    p_ekf_r = np.logspace(-4, 1, 8)

    p_ukf_q = p_ekf_q
    p_ukf_r = p_ekf_r

    p_lkf_q = p_ekf_q
    p_lkf_r = p_ekf_r

    p_sogi_k = [1.0, 1.414, 2.0]
    p_sogi_g = [50, 100, 200, 300]

    p_rls_lam = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]
    p_rls_win = [50, 100, 200, 500]
    rls_decim = 50

    p_teager_win = [10, 20, 30, 40, 50]
    p_tft_win = [2, 3, 4, 6]

    p_vff_lam_min = [0.90, 0.95, 0.98, 0.99]
    p_vff_Ka = [1.0, 2.0, 5.0]
    vff_win_smooth = 20
    vff_decim = 50
    vff_lam_max = 0.9995

    p_koopman_win = [10, 40, 80, 160, 333, 800, 2000, 5000]
    kalman_smooth_win = 10

    grids = {
        "p_ipdft": p_ipdft,
        "p_pll_kp": p_pll_kp,
        "p_pll_ki": p_pll_ki,
        "p_ekf_q": p_ekf_q,
        "p_ekf_r": p_ekf_r,
        "p_ukf_q": p_ukf_q,
        "p_ukf_r": p_ukf_r,
        "p_lkf_q": p_lkf_q,
        "p_lkf_r": p_lkf_r,
        "p_sogi_k": p_sogi_k,
        "p_sogi_g": p_sogi_g,
        "p_rls_lam": p_rls_lam,
        "p_rls_win": p_rls_win,
        "rls_decim": rls_decim,
        "p_teager_win": p_teager_win,
        "p_tft_win": p_tft_win,
        "p_vff_lam_min": p_vff_lam_min,
        "p_vff_Ka": p_vff_Ka,
        "vff_win_smooth": vff_win_smooth,
        "vff_decim": vff_decim,
        "vff_lam_max": vff_lam_max,
        "p_koopman_win": p_koopman_win,
        "kalman_smooth_win": kalman_smooth_win,
    }

    methods = [
        "IpDFT",
        "PLL",
        "EKF",
        "EKF2",
        "SOGI",
        "RLS",
        "Teager",
        "TFT",
        "RLS-VFF",
        "UKF",
        "LKF",
        "Koopman-RKDPmu",
        "PI-GRU",
    ]

    print(
        f"Running Q1 Monte Carlo Benchmark "
        f"({len(methods)} methods) @ {FS_DSP} Hz | "
        f"train={MC_CFG['n_train_seeds']} test={MC_CFG['n_test_seeds']}"
    )
    print("-" * 80)

    # =============================================================
    # LOOP escenarios
    # =============================================================
    for sc_name, (t_phys, v_ana, f_true, meta) in signals.items():
        print(f">> Scenario: {sc_name}")

        v_dsp = v_ana[::RATIO]
        f_target = f_true[::RATIO]
        t_dsp = t_phys[::RATIO]

        # Landscapes en señal base (visualización)
        generate_pll_landscape(v_dsp, f_target, p_pll_kp, p_pll_ki, sc_name)
        generate_ekf_landscape(v_dsp, f_target, p_ekf_q, p_ekf_r, sc_name)
        generate_rls_landscape(v_dsp, f_target, p_rls_lam, p_rls_win, sc_name)

        scenario_mc, results_map_rep = run_mc_for_scenario(
            sc_name=sc_name,
            t_dsp=t_dsp,
            v_dsp=v_dsp,
            f_target=f_target,
            meta=meta,
            methods=methods,
            grids=grids,
        )

        # Guardar summary MC por escenario
        out_path = os.path.join(RESULTS_MC_DIR, f"{sc_name}__mc_summary.json")
        with open(out_path, "w") as f:
            json.dump(scenario_mc, f, indent=2)

        # =============================================================
        # Construir json_export compatible con tus plots globales
        # Guardamos valores principales como mediana (p50) de test_agg
        # + guardamos stats completos en llaves *_STAT para tablas Q1
        # =============================================================
        json_export["results"][sc_name] = {"scenario_description": meta, "methods": {}}

        for method in methods:
            test_agg = scenario_mc["methods"][method]["test_agg"]

            # principal = p50 (mediana)
            compact = {}
            for metric_name, stat in test_agg.items():
                # si tu plotting espera floats, usamos p50
                compact[metric_name] = stat.get("p50", stat.get("median", stat.get("mean", 0.0)))
                compact[f"{metric_name}_STAT"] = stat  # full stats

            compact["optimal_params"] = scenario_mc["methods"][method]["tuning_robust"].get("label", "")
            compact["MC_REP_SEED"] = scenario_mc["methods"][method]["representative_seed"]
            json_export["results"][sc_name]["methods"][method] = compact

        # =============================================================
        # Plot por escenario (representative run)
        # Ojo: aquí graficamos SOLO el segmento TEST
        # =============================================================
        n = len(v_dsp)
        tune_idx, test_idx = split_indices(n, MC_CFG["tune_frac"])
        t_test = t_dsp[test_idx]
        f_test = f_target[test_idx]
        save_plots(sc_name, t_test, f_test, results_map_rep)

        print(f"   Saved MC summary: {out_path}")

    # =============================================================
    # GLOBAL SUMMARIES (tu plotting.py existente)
    # =============================================================
    save_metrics_summary(json_export)
    save_pareto_plots(json_export)
    save_risk_plots(json_export)

    # Guardar JSON global compatible
    with open(f"{OUTPUT_DIR}/benchmark_results_mc_compat.json", "w") as f:
        json.dump(json_export, f, indent=2)

    # Guardar JSON global MC (full)
    full_mc_path = os.path.join(RESULTS_MC_DIR, "benchmark_results_mc.json")
    with open(full_mc_path, "w") as f:
        json.dump(json_export, f, indent=2)

    print("-" * 80)
    print(f"Benchmark Complete.")
    print(f"  Figures/compat JSON: {OUTPUT_DIR}")
    print(f"  Raw per-seed: {RESULTS_RAW_DIR}")
    print(f"  MC summaries: {RESULTS_MC_DIR}")


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    run_benchmark()
