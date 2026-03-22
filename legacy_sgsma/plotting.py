# plotting.py
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# 0. OUTPUT DIR + IEEE STYLE SOLO PARA FIGURAS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures_estimatores_benchmark_test5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _reset_output_dir():
    """Clean start for each run. Called explicitly from main, not at import time."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Matplotlib Configuration for IEEE Conference Papers
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,            # Standard IEEE caption/label size
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,      # Slightly smaller for legends
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": (3.5, 2.8),  # Default to single column width
    "lines.linewidth": 1.0,    # Thinner lines for smaller figures
    "grid.alpha": 0.4,
    "savefig.dpi": 600,        # High resolution for print
    "savefig.bbox": "tight",
    "figure.autolayout": True  # Auto-adjust layout to prevent clipping
})

# Umbral de disparo que usan tanto métricas como plots
TRIP_THRESHOLD = 0.5  # Hz


# =============================================================
# 1. ANOTACIONES IEEE PARA ESCENARIOS
# =============================================================

def add_ieee_annotations(ax, scenario):
    if scenario == "IEEE_Mag_Step":
        ax.annotate(
            "Step t=0.5s",
            xy=(0.5, 60.0),
            xytext=(0.6, 60.5),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=9
        )
    elif scenario == "IEEE_Freq_Ramp":
        ax.annotate(
            "Ramp Start",
            xy=(0.3, 60.0),
            xytext=(0.1, 61.5),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=9
        )
    elif scenario == "IEEE_Modulation":
        ax.annotate(
            "AM/FM Mod",
            xy=(0.5, 60.0),
            xytext=(0.6, 60.1),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=9
        )
    elif scenario == "IBR_Nightmare":
        ax.annotate(
            "Phase Jump (60°)",
            xy=(0.7, 60.0),
            xytext=(0.85, 63.0),
            arrowprops=dict(facecolor="red", arrowstyle="->"),
            fontsize=9,
            color="red"
        )


# =============================================================
# 2. PLOTS POR ESCENARIO (TRACES + ERRORES + ZOOM)
# =============================================================

def _zoom_cfg(sc_name):
    """
    Devuelve (t_min, t_max, y_min, y_max, e_max) para la figura de zoom.
    e_max = límite superior para |error| en el zoom.
    """
    if sc_name == "IEEE_Mag_Step":
        return 0.0, 0.15, 58.5, 60.5, 1.0
    if sc_name == "IEEE_Freq_Ramp":
        return 0.25, 1.05, 59.0, 64.0, 2.0
    if sc_name == "IEEE_Modulation":
        return 0.0, 0.5, 59.0, 61.0, 1.0
    if sc_name == "IBR_Nightmare":
        return 0.65, 0.95, 58.0, 70.0, 5.0
    if sc_name == "IBR_MultiEvent":
        return 0.9, 3.0, 50.0, 70.0, 10.0
    return 0.0, 1.5, 55.0, 65.0, 3.0


def save_plots(sc_name, t, f_true, results_map):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "IpDFT": "blue",
        "PLL": "green",
        "EKF": "red",
        "EKF2": "brown",
        "SOGI": "magenta",
        "RLS": "cyan",
        "Teager": "orange",
        "TFT": "purple",
        "RLS-VFF": "darkcyan",
        "UKF": "darkred",
        "LKF": "slategray",
        "Koopman-RKDPmu": "olive",
        "PI-GRU": "black",
    }

    order = [
        "Teager",
        "RLS",
        "RLS-VFF",
        "IpDFT",
        "PLL",
        "SOGI",
        "TFT",
        "EKF",
        "EKF2",
        "UKF",
        "Koopman-RKDPmu",
        "PI-GRU",
    ]

    good_methods = [
        "PLL", "SOGI", "TFT",
        "EKF", "EKF2", "UKF",
        "RLS", "RLS-VFF", "Koopman-RKDPmu",
        "PI-GRU",
    ]

    # =====================================================
    # 1) Figura global (toda la señal)
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(t, f_true, "k--", linewidth=1.2, label="True", alpha=0.5)

    for method in order:
        if method in results_map:
            data = results_map[method]
            if method in ["EKF", "EKF2", "TFT", "PLL", "UKF", "Koopman-RKDPmu", "PI-GRU"]:
                lw = 1.2
                alpha = 0.9
                z = 3
            elif method in ["Teager", "RLS", "RLS-VFF"]:
                lw = 0.6
                alpha = 0.18
                z = 1
            else:
                lw = 0.8
                alpha = 0.6
                z = 2

            ax1.plot(
                t, data["trace"],
                color=colors.get(method, "gray"),
                label=method,
                linewidth=lw,
                alpha=alpha,
                zorder=z
            )

    ax1.set_ylabel("Freq [Hz]")
    ax1.set_title(f"{sc_name} Analysis")
    ax1.grid(True, which="both", alpha=0.4)
    ymin, ymax = ax1.get_ylim()
    Y_MIN = 55.0
    Y_MAX = 65.0
    ymin = max(ymin, Y_MIN)
    ymax = min(ymax, Y_MAX)
    ax1.set_ylim(ymin, ymax)
    add_ieee_annotations(ax1, sc_name)
    ax1.legend(loc="upper right", ncol=1, fontsize=5, framealpha=0.8)

    for method in order:
        if method in results_map:
            data = results_map[method]
            err = np.abs(data["trace"] - f_true)
            if method in ["EKF", "EKF2", "TFT", "PLL", "UKF", "Koopman-RKDPmu", "PI-GRU"]:
                alpha = 0.8
                lw = 0.9
            elif method in ["Teager", "RLS", "RLS-VFF"]:
                alpha = 0.25
                lw = 0.6
            else:
                alpha = 0.5
                lw = 0.8

            ax2.plot(
                t, err,
                color=colors.get(method, "gray"),
                linewidth=lw,
                alpha=alpha,
                label=method
            )

    ax2.axhline(
        TRIP_THRESHOLD,
        color="black",
        linestyle=":",
        label="Trip Threshold",
        linewidth=1.0
    )
    ax2.set_ylabel("|Error| [Hz]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, which="both", alpha=0.4)

    if sc_name == "IBR_Nightmare":
        ax2.set_ylim(0, 5.0)
    elif sc_name == "IBR_MultiEvent":
        ax2.set_ylim(0, 10.0)
    else:
        ax2.set_ylim(0, 3.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sc_name}_ALL_METHODS.png", dpi=600)
    plt.close(fig)

    # =====================================================
    # 2) Figuras individuales (una por método)
    # =====================================================
    for method in order:
        if method in results_map:
            fig = plt.figure(figsize=(3.5, 2.5))
            plt.plot(t, f_true, "k--", linewidth=1.2, label="True", alpha=0.5)

            alpha = 0.9 if method in good_methods else 0.5
            lw = 1.0 if method in good_methods else 0.7

            plt.plot(
                t, results_map[method]["trace"],
                color=colors.get(method, "gray"),
                linewidth=lw,
                label=method,
                alpha=alpha
            )
            plt.title(f"{sc_name}: {method}")
            plt.xlabel("Time [s]")
            plt.ylabel("Freq [Hz]")
            plt.grid(True, which="both", alpha=0.4)
            plt.ylim(55.0, 65.0)
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{sc_name}_{method}_Optimized.png", dpi=600)
            plt.close(fig)

    # =====================================================
    # 3) Figura de ZOOM (freq + error, solo métodos buenos)
    # =====================================================
    t_min, t_max, y_min, y_max, e_max = _zoom_cfg(sc_name)
    mask = (t >= t_min) & (t <= t_max)

    figz, (axz1, axz2) = plt.subplots(
        2, 1, figsize=(3.5, 3.8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axz1.plot(
        t[mask], f_true[mask],
        "k--", lw=1.2, alpha=0.5, label="True"
    )

    for method in good_methods:
        if method in results_map:
            data = results_map[method]
            label = f"{method} (Tset={data['SETTLING']*1000:.0f} ms)"
            axz1.plot(
                t[mask], data["trace"][mask],
                color=colors.get(method, "gray"),
                lw=1.2, alpha=0.95,
                label=label
            )

    axz1.set_ylabel("Freq [Hz]")
    axz1.set_title("Transient Recovery (Zoom)")
    axz1.grid(True, which="both", alpha=0.4)
    axz1.set_ylim(y_min, y_max)
    axz1.legend(ncol=1, fontsize=6, loc="best")

    for method in good_methods:
        if method in results_map:
            data = results_map[method]
            err = np.abs(data["trace"] - f_true)
            axz2.plot(
                t[mask], err[mask],
                color=colors.get(method, "gray"),
                lw=1.0, alpha=0.9,
                label=method
            )

    axz2.axhline(
        TRIP_THRESHOLD,
        color="black", linestyle=":",
        lw=1.0, label="Trip"
    )
    axz2.set_ylabel("|Error| [Hz]")
    axz2.set_xlabel("Time [s]")
    axz2.grid(True, which="both", alpha=0.4)
    axz2.set_ylim(0, e_max)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sc_name}_ZOOM_FREQ_ERROR.png", dpi=600)
    plt.close(figz)


# =============================================================
# 3. RESUMEN GLOBAL DE MÉTRICAS (BARRAS)
# =============================================================

def save_metrics_summary(json_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = list(json_data["results"].keys())
    first_scenario = scenarios[0]
    methods = list(json_data["results"][first_scenario]["methods"].keys())
    n_groups = len(scenarios)
    bar_width = 0.1

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    index = np.arange(n_groups)

    colors = {
        "IpDFT": "blue",
        "PLL": "green",
        "EKF": "red",
        "EKF2": "brown",
        "SOGI": "magenta",
        "RLS": "cyan",
        "Teager": "orange",
        "TFT": "purple",
        "RLS-VFF": "darkcyan",
        "UKF": "darkred",
        "Koopman-RKDPmu": "olive",
        "PI-GRU": "black",
    }

    scenario_labels_map = {
        "IEEE_Mag_Step": "Mag Step",
        "IEEE_Freq_Ramp": "Freq Ramp",
        "IEEE_Modulation": "Modulation",
        "IBR_Nightmare": "Nightmare",
        "IBR_MultiEvent": "Multi-Event"
    }
    xtick_labels = [scenario_labels_map.get(sc, sc) for sc in scenarios]

    for i, method in enumerate(methods):
        rmses = [json_data["results"][sc]["methods"][method]["RMSE"] for sc in scenarios]
        ax.bar(
            index + i * bar_width, rmses,
            bar_width, alpha=0.8,
            color=colors.get(method, "gray"),
            label=method
        )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("RMSE [Hz] (Log Scale)")
    ax.set_title("RMSE Comparison by Method and Scenario")
    ax.set_xticks(index + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_yscale("log")
    ax.legend(ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.grid(True, which="both", axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "SUMMARY_RMSE_METRICS.png", dpi=600)
    plt.close(fig)


# =============================================================
# 4. PARETO Y RISK VISUALIZATION
# =============================================================

def compute_pareto_front(points):
    """
    points: list of (rmse, complexity, name)
    Returns indices of non-dominated points (minimize both).
    """
    pareto_idx = []
    for i, (ri, ci, _) in enumerate(points):
        dominated = False
        for j, (rj, cj, _) in enumerate(points):
            if j == i:
                continue
            if (rj <= ri and cj <= ci) and (rj < ri or cj < ci):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return pareto_idx


def save_pareto_plots(json_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "IpDFT": "blue",
        "PLL": "green",
        "EKF": "red",
        "EKF2": "brown",
        "SOGI": "magenta",
        "RLS": "cyan",
        "Teager": "orange",
        "TFT": "purple",
        "RLS-VFF": "darkcyan",
        "UKF": "darkred",
        "LKF": "slategray",
        "Koopman-RKDPmu": "olive",
        "PI-GRU": "black",
    }

    for sc_name, sc_data in json_data["results"].items():
        methods = sc_data["methods"]
        points = []
        for m_name, vals in methods.items():
            rmse = vals["RMSE"]
            comp = vals.get("TIME_PER_SAMPLE_US", 0.0)
            points.append((rmse, comp, m_name))

        pareto_ids = compute_pareto_front(points)

        fig = plt.figure(figsize=(3.5, 2.8))
        for idx, (rmse, comp, name) in enumerate(points):
            if comp <= 0:
                continue
            if idx in pareto_ids:
                plt.scatter(
                    comp, rmse, s=40,
                    edgecolors="black",
                    color=colors.get(name, "gray"),
                    label=f"{name} (Pareto)"
                )
            else:
                plt.scatter(
                    comp, rmse, s=20,
                    color=colors.get(name, "gray"),
                    alpha=0.6,
                    label=name
                )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Time per sample [µs]")
        plt.ylabel("RMSE [Hz]")
        plt.title(f"Pareto Trade-off (Accuracy vs Complexity)\n{sc_name}")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=6, loc="best")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{sc_name}_PARETO_RMSE_COMPLEXITY.png", dpi=600)
        plt.close(fig)


def save_risk_plots(json_data):
    """
    Plot risk metrics: total time |e|>0.5 Hz and max continuous excursion.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "IpDFT": "blue",
        "PLL": "green",
        "EKF": "red",
        "EKF2": "brown",
        "SOGI": "magenta",
        "RLS": "cyan",
        "Teager": "orange",
        "TFT": "purple",
        "RLS-VFF": "darkcyan",
        "UKF": "darkred",
        "Koopman-RKDPmu": "olive",
        "PI-GRU": "black",
    }

    for sc_name, sc_data in json_data["results"].items():
        methods = sc_data["methods"]
        m_names = list(methods.keys())
        trip_times = [methods[m]["TRIP_TIME_0p5"] for m in m_names]
        max_cont = [methods[m]["MAX_CONTIGUOUS_0p5"] for m in m_names]

        x = np.arange(len(m_names))
        width = 0.35

        fig = plt.figure(figsize=(3.5, 2.5))
        plt.bar(x, trip_times, width, color=[colors.get(m, "gray") for m in m_names])
        plt.xticks(x, m_names, rotation=45, ha="right")
        plt.ylabel("Total time |e| > 0.5 Hz [s]")
        plt.title(f"Risk of Spurious Trips (Total Time)\n{sc_name}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{sc_name}_RISK_TOTAL_TRIPTIME.png", dpi=600)
        plt.close(fig)

        fig = plt.figure(figsize=(3.5, 2.5))
        plt.bar(x, max_cont, width, color=[colors.get(m, "gray") for m in m_names])
        plt.xticks(x, m_names, rotation=45, ha="right")
        plt.ylabel("Max contiguous |e| > 0.5 Hz [s]")
        plt.title(f"Risk of Spurious Trips (Max Burst)\n{sc_name}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{sc_name}_RISK_MAX_BURST.png", dpi=600)
        plt.close(fig)


# =============================================================
# 5. HYPERPARAMETER LANDSCAPES
# =============================================================

def generate_pll_landscape(v, f, kp_vals, ki_vals, sc_name):
    from estimators import StandardPLL, calculate_metrics

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rmse_grid = np.zeros((len(kp_vals), len(ki_vals)))
    for i, kp in enumerate(kp_vals):
        for j, ki in enumerate(ki_vals):
            algo = StandardPLL(kp, ki)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.maf_win)
            rmse_grid[i, j] = m["RMSE"]

    fig = plt.figure(figsize=(3.5, 2.8))
    im = plt.imshow(
        rmse_grid,
        origin="lower",
        aspect="auto",
        extent=[ki_vals[0], ki_vals[-1], kp_vals[0], kp_vals[-1]],
        interpolation="nearest"
    )
    plt.colorbar(im, label="RMSE [Hz]")
    plt.xlabel("Ki")
    plt.ylabel("Kp")
    plt.title(f"PLL Hyperparameter Landscape (RMSE)\n{sc_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sc_name}_PLL_HYPERMAP.png", dpi=600)
    plt.close(fig)


def generate_ekf_landscape(v, f, q_vals, r_vals, sc_name):
    from estimators import ClassicEKF, calculate_metrics

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rmse_grid = np.zeros((len(q_vals), len(r_vals)))
    for i, q in enumerate(q_vals):
        for j, r in enumerate(r_vals):
            algo = ClassicEKF(q, r)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=1)
            rmse_grid[i, j] = m["RMSE"]

    fig = plt.figure(figsize=(3.5, 2.8))
    im = plt.imshow(
        rmse_grid,
        origin="lower",
        aspect="auto",
        extent=[
            np.log10(r_vals[0]), np.log10(r_vals[-1]),
            np.log10(q_vals[0]), np.log10(q_vals[-1])
        ],
        interpolation="nearest"
    )
    plt.colorbar(im, label="RMSE [Hz]")
    plt.xlabel("log10(R)")
    plt.ylabel("log10(Q)")
    plt.title(f"EKF Hyperparameter Landscape (RMSE)\n{sc_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sc_name}_EKF_HYPERMAP.png", dpi=600)
    plt.close(fig)


def generate_rls_landscape(v, f, lam_vals, win_vals, sc_name):
    from estimators import RLS_Estimator, calculate_metrics

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rmse_grid = np.zeros((len(lam_vals), len(win_vals)))
    for i, lam in enumerate(lam_vals):
        for j, w in enumerate(win_vals):
            algo = RLS_Estimator(lam, w)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.smooth_win)
            rmse_grid[i, j] = m["RMSE"]

    fig = plt.figure(figsize=(3.5, 2.8))
    im = plt.imshow(
        rmse_grid,
        origin="lower",
        aspect="auto",
        extent=[win_vals[0], win_vals[-1], lam_vals[0], lam_vals[-1]],
        interpolation="nearest"
    )
    plt.colorbar(im, label="RMSE [Hz]")
    plt.xlabel("Smoothing window [samples]")
    plt.ylabel("Lambda")
    plt.title(f"RLS Hyperparameter Landscape (RMSE)\n{sc_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sc_name}_RLS_HYPERMAP.png", dpi=600)
    plt.close(fig)


# =============================================================
# 6. COMPLIANCE HEATMAP  (Fig. 2g in paper)
# =============================================================

PASS_RMSE_HZ = 0.05   # Hz
PASS_PEAK_HZ = 0.50   # Hz
PASS_TRIP_S = 0.10    # s


def _passes(method_metrics):
    """Return True if method_metrics satisfies all Pass criteria."""
    return (
        method_metrics.get("RMSE", 1e9) < PASS_RMSE_HZ
        and method_metrics.get("MAX_PEAK", 1e9) < PASS_PEAK_HZ
        and method_metrics.get("TRIP_TIME_0p5", 1e9) < PASS_TRIP_S
    )


def save_compliance_heatmap(json_data):
    """
    Generate a Pass/Fail compliance heatmap across all methods and scenarios.
    Saves to OUTPUT_DIR/COMPLIANCE_HEATMAP.png
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = json_data["results"]
    scenarios = list(results.keys())

    method_set = []
    for sc_data in results.values():
        for m in sc_data["methods"]:
            if m not in method_set:
                method_set.append(m)

    preferred_order = [
        "EKF2", "UKF", "EKF",
        "PLL", "SOGI",
        "IpDFT", "TFT",
        "RLS-VFF", "RLS", "LKF",
        "Koopman-RKDPmu", "Teager",
        "PI-GRU",
    ]
    methods_ordered = [m for m in preferred_order if m in method_set]
    methods_ordered += [m for m in method_set if m not in methods_ordered]

    sc_labels = {
        "IEEE_Mag_Step": "Step",
        "IEEE_Freq_Ramp": "Ramp",
        "IEEE_Modulation": "Mod",
        "IBR_Nightmare": "Isl",
        "IBR_MultiEvent": "Multi",
    }

    n_methods = len(methods_ordered)
    n_scenarios = len(scenarios)

    fig, ax = plt.subplots(figsize=(3.5, 0.38 * n_methods + 1.0))

    for col_idx, sc in enumerate(scenarios):
        sc_methods = results[sc].get("methods", {})
        for row_idx, method in enumerate(methods_ordered):
            if method in sc_methods:
                passed = _passes(sc_methods[method])
            else:
                passed = None

            if passed is True:
                facecolor = "#2ca02c"
                label_txt = "P"
            elif passed is False:
                facecolor = "#d62728"
                label_txt = "F"
            else:
                facecolor = "#aaaaaa"
                label_txt = "N/A"

            rect = plt.Rectangle(
                [col_idx - 0.5, row_idx - 0.5], 1.0, 1.0,
                color=facecolor, alpha=0.85
            )
            ax.add_patch(rect)
            ax.text(
                col_idx, row_idx, label_txt,
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white"
            )

    ax.set_xlim(-0.5, n_scenarios - 0.5)
    ax.set_ylim(-0.5, n_methods - 0.5)
    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels([sc_labels.get(s, s) for s in scenarios], fontsize=7)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(methods_ordered, fontsize=7)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(
        f"Compliance Heatmap (P=Pass / F=Fail)\n"
        f"Pass: RMSE<{PASS_RMSE_HZ} Hz, Peak<{PASS_PEAK_HZ} Hz, "
        f"TripTime<{PASS_TRIP_S} s",
        fontsize=7, pad=18
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "COMPLIANCE_HEATMAP.png", dpi=600)
    plt.savefig(OUTPUT_DIR / "COMPLIANCE_HEATMAP.pdf")
    plt.close(fig)
    print("[OK] COMPLIANCE_HEATMAP saved")