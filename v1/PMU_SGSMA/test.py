# test_rls_sanity.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import platform
import sys

from estimators import (
    FS_DSP,
    FS_PHYSICS,
    RATIO,
    get_test_signals,
    RLS_Estimator,
    RLS_VFF_Estimator,
    calculate_metrics,
)

# ---------------------------------------------
# Config
# ---------------------------------------------
OUTPUT_DIR = "figures_rls_sanity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 8,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "legend.frameon": False,
    }
)


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def run_single_case(name, t_dsp, v_dsp, f_true, rls_params, vff_params):
    """
    Ejecuta RLS y RLS-VFF sobre un escenario y devuelve métricas y trazas.
    """
    results = {}

    # ----- RLS clásico -----
    lam_rls, win_rls = rls_params
    rls = RLS_Estimator(lam=lam_rls, win_smooth=win_rls)
    tr_rls = np.array([rls.step(x) for x in v_dsp])
    m_rls = calculate_metrics(
        tr_rls, f_true, exec_time=0.0, structural_samples=rls.smooth_win
    )
    m_rls["params"] = {"lam": lam_rls, "win": win_rls}
    results["RLS"] = {"metrics": m_rls, "trace": tr_rls}

    # ----- RLS-VFF -----
    lam_min, lam_max, alpha_vff, win_vff = vff_params
    rls_vff = RLS_VFF_Estimator(
        lam_min=lam_min,
        lam_max=lam_max,
        alpha=alpha_vff,
        win_smooth=win_vff,
    )
    tr_vff = np.array([rls_vff.step(x) for x in v_dsp])
    m_vff = calculate_metrics(
        tr_vff, f_true, exec_time=0.0, structural_samples=rls_vff.smooth_win
    )
    m_vff["params"] = {
        "lam_min": lam_min,
        "lam_max": lam_max,
        "alpha": alpha_vff,
        "win": win_vff,
    }
    results["RLS-VFF"] = {"metrics": m_vff, "trace": tr_vff}

    # ----- Plot sencillo -----
    fig, axes = plt.subplots(
        2, 1, figsize=(4.0, 3.4), sharex=True, constrained_layout=True
    )

    # Freq
    axes[0].plot(t_dsp, f_true, "k--", label="True")
    axes[0].plot(t_dsp, tr_rls, label="RLS")
    axes[0].plot(t_dsp, tr_vff, label="RLS-VFF")
    axes[0].set_ylabel("Freq [Hz]")
    axes[0].set_title(name)
    axes[0].legend(loc="best")

    # Error |e|
    axes[1].plot(t_dsp, np.abs(tr_rls - f_true), label="|e| RLS")
    axes[1].plot(t_dsp, np.abs(tr_vff - f_true), label="|e| RLS-VFF")
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=0.6)
    axes[1].set_ylabel("Error [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="best")

    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_RLS_vs_VFF.png"))
    plt.close(fig)

    return results


def main():
    # --------------------------
    # 0. Meta-info para JSON
    # --------------------------
    export = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "description": (
                "Sanity-check ligero de RLS y RLS-VFF en tres escenarios: "
                "Sine60, IEEE_Mag_Step, IEEE_Modulation."
            ),
            "pc_hostname": platform.node(),
            "machine_arch": platform.machine(),
            "cpu_processor": platform.processor(),
            "os_platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "fs_physics_hz": FS_PHYSICS,
            "fs_dsp_hz": FS_DSP,
            "downsampling_ratio": RATIO,
        },
        "results": {},
    }

    # --------------------------
    # 1. Escenario Seno 60 Hz
    # --------------------------
    dur = 1.5  # s
    t_dsp = np.arange(0, dur, 1.0 / FS_DSP)
    v_sine = np.sin(2.0 * np.pi * 60.0 * t_dsp)
    f_true_sine = np.ones_like(t_dsp) * 60.0

    # Parámetros "razonables" para el sanity test
    rls_params = (0.99, 100)  # (lambda, win_smooth)
    vff_params = (0.95, 0.9999, 5.0, 40)  # (lam_min, lam_max, alpha, win_smooth)

    print("\n=== SCENARIO: Sine60 ===")
    res_sine = run_single_case(
        "Sine60", t_dsp, v_sine, f_true_sine, rls_params, vff_params
    )
    for m_name, data in res_sine.items():
        m = data["metrics"]
        print(
            f"{m_name:8s} | RMSE={m['RMSE']:.4f} Hz | "
            f"MAX_PEAK={m['MAX_PEAK']:.2f} Hz | "
            f"TRIP_TIME_0p5={m['TRIP_TIME_0p5']:.4f} s"
        )

    export["results"]["Sine60"] = {
        "methods": {k: {"metrics": v["metrics"]} for k, v in res_sine.items()}
    }

    # --------------------------
    # 2. Escenarios IEEE
    # --------------------------
    signals = get_test_signals()

    for sc_name in ["IEEE_Mag_Step", "IEEE_Modulation"]:
        t_phys, v_ana, f_true_phys, meta = signals[sc_name]
        t_dsp = t_phys[::RATIO]
        v_dsp = v_ana[::RATIO]
        f_true = f_true_phys[::RATIO]

        print(f"\n=== SCENARIO: {sc_name} ===")
        res = run_single_case(sc_name, t_dsp, v_dsp, f_true, rls_params, vff_params)
        for m_name, data in res.items():
            m = data["metrics"]
            print(
                f"{m_name:8s} | RMSE={m['RMSE']:.4f} Hz | "
                f"MAX_PEAK={m['MAX_PEAK']:.2f} Hz | "
                f"TRIP_TIME_0p5={m['TRIP_TIME_0p5']:.4f} s"
            )

        export["results"][sc_name] = {
            "scenario_description": meta,
            "methods": {k: {"metrics": v["metrics"]} for k, v in res.items()},
        }

    # --------------------------
    # Guardar JSON resumen
    # --------------------------
    json_path = os.path.join(OUTPUT_DIR, "rls_sanity_results.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=4)

    print(f"\nSanity-check completo. Figuras y JSON en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
