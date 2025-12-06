#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main2.py - LIGHT benchmark:
    - Métodos: RLS y RLS-VFF únicamente
    - Escenarios: IEEE_Mag_Step, IEEE_Modulation
    - Usa directamente las funciones de estimators.py:
        * get_test_signals
        * RLS_Estimator
        * RLS_VFF_Estimator
        * calculate_metrics
        * tune_rls
        * tune_vff_rls
"""

import numpy as np
import time
import sys
import json
import platform
import datetime

from estimators import (
    SEED,
    FS_PHYSICS,
    FS_DSP,
    RATIO,
    get_test_signals,
    RLS_Estimator,
    RLS_VFF_Estimator,
    calculate_metrics,
    tune_rls,
    tune_vff_rls,
)

from plotting import (
    OUTPUT_DIR,
    save_plots,
    save_metrics_summary,
)


# =============================================================
# MAIN LIGHT
# =============================================================
def run_benchmark_light():

    signals = get_test_signals()

    json_export = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "description": (
                "LIGHT benchmark: RLS & RLS-VFF únicamente, "
                "solo escenarios IEEE_Mag_Step y IEEE_Modulation."
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

    # Grids pequeños para LIGHT run
    p_rls_lam = [0.95]
    p_rls_win = [40]

    p_vff_lam_min = [0.90, 0.95]
    p_vff_alpha = [1.0, 2.0, 5.0]
    vff_win_smooth = 20
    decim_common = 50  # debe coincidir con tune_rls / tune_vff_rls

    print(f"Running LIGHT Benchmark (RLS & RLS-VFF only) @ {FS_DSP} Hz ...")
    print("-" * 80)

    scenarios_light = ["IEEE_Mag_Step", "IEEE_Modulation"]

    for sc_name in scenarios_light:
        if sc_name not in signals:
            print(f"[WARN] Scenario {sc_name} no está en get_test_signals(), se omite.")
            continue

        print(f">> Processing SCENARIO (LIGHT): {sc_name}")

        t_phys, v_ana, f_true, meta = signals[sc_name]

        # Downsample a 10 kHz
        v_dsp = v_ana[::RATIO]
        f_target = f_true[::RATIO]
        t_dsp = t_phys[::RATIO]

        results_map = {}
        json_export["results"][sc_name] = {
            "scenario_description": meta,
            "methods": {},
        }

        # =========================================================
        # 1) RLS (baseline)
        # =========================================================
        p_str_rls, (lam_rls, win_rls) = tune_rls(
            v_dsp,
            f_target,
            lam_vals=p_rls_lam,
            win_vals=p_rls_win,
            sc_name=sc_name,
        )

        t0 = time.process_time()
        algo_rls = RLS_Estimator(
            lam=lam_rls,
            win_smooth=win_rls,
            decim=decim_common,
        )
        tr_rls = np.array([algo_rls.step(x) for x in v_dsp])
        exec_t_rls = time.process_time() - t0

        m_rls = calculate_metrics(
            tr_rls,
            f_target,
            exec_t_rls,
            structural_samples=algo_rls.smooth_win * algo_rls.decim,
        )
        m_rls["optimal_params"] = p_str_rls
        results_map["RLS"] = {**m_rls, "trace": tr_rls}

        # =========================================================
        # 2) RLS-VFF (usa tune_vff_rls de estimators.py)
        # =========================================================
        p_str_vff, (lam_min_vff, Ka_vff) = tune_vff_rls(
            v_dsp,
            f_target,
            lam_min_vals=p_vff_lam_min,
            alpha_vals=p_vff_alpha,
            sc_name=sc_name,
        )

        t0 = time.process_time()
        algo_vff = RLS_VFF_Estimator(
            lam_min=lam_min_vff,
            lam_max=0.9995,
            Ka=Ka_vff,       # Ka controla la ventana exponencial
            Kb=None,         # Kb = Ka internamente
            win_smooth=vff_win_smooth,
            decim=decim_common,
        )
        tr_vff = np.array([algo_vff.step(x) for x in v_dsp])
        exec_t_vff = time.process_time() - t0

        m_vff = calculate_metrics(
            tr_vff,
            f_target,
            exec_t_vff,
            structural_samples=algo_vff.smooth_win * algo_vff.decim,
        )
        m_vff["optimal_params"] = p_str_vff
        results_map["RLS-VFF"] = {**m_vff, "trace": tr_vff}

        # =========================================================
        # PLOTS por escenario
        # =========================================================
        save_plots(sc_name, t_dsp, f_target, results_map)

        # Export JSON por escenario (sin el trace)
        for method, vals in results_map.items():
            json_export["results"][sc_name]["methods"][method] = {
                key: val for key, val in vals.items() if key != "trace"
            }
            print(
                f"   [{method:<10}] RMSE={vals['RMSE']:.4f} | "
                f"Peak={vals['MAX_PEAK']:.2f} | "
                f"TripTime={vals['TRIP_TIME_0p5']:.4f}s | "
                f"CPU={vals['TIME_PER_SAMPLE_US']:.2e}µs"
            )

    # =============================================================
    # GLOBAL SUMMARY
    # =============================================================
    save_metrics_summary(json_export)

    with open(f"{OUTPUT_DIR}/benchmark_results_light.json", "w") as f:
        json.dump(json_export, f, indent=4)

    print(
        f"\nLIGHT Benchmark Complete. "
        f"Results saved in {OUTPUT_DIR}/benchmark_results_light.json"
    )


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    run_benchmark_light()
