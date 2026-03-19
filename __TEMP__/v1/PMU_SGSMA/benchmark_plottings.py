#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_plotting.py

Post-processing Q1/IEEE-style plotting for the OpenFreqBench benchmark.

Reads raw JSON files produced in `results_raw/<scenario>/<scenario>__<method>.json`,
aggregates metrics and generates publication-grade figures.

Author: Jorge L. Mayorga (with ChatGPT as plotting slave)
"""

import os
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ============================================================================
# 0. GLOBAL PATHS & STYLE (IEEE-like)
# ============================================================================

RESULTS_RAW_DIR = "results_raw"
FIG_DIR = "figures_from_raw"
os.makedirs(FIG_DIR, exist_ok=True)

# IEEE-ish single / double column sizes (inches)
FIGSIZE_SINGLE = (3.5, 2.2)  # one-column
FIGSIZE_DOUBLE = (7.2, 2.6)  # two-column wide but not too tall

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.4,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "axes.grid": True,
        "figure.autolayout": False,
    }
)

# Ordered scenario keys and pretty labels (Ramp/Modulation swap already applied)
SCENARIO_ORDER = [
    "IEEE_Mag_Step",
    "IEEE_Modulation",  # swapped
    "IEEE_Freq_Ramp",  # swapped
    "IBR_Nightmare",
    "IBR_MultiEvent",
]
SCENARIO_LABELS = {
    "IEEE_Mag_Step": "Mag Step",
    "IEEE_Freq_Ramp": "Freq Ramp",
    "IEEE_Modulation": "Modulation",
    "IBR_Nightmare": "IBR Nightmare",
    "IBR_MultiEvent": "IBR MultiEvent",
}

# Ordered methods and colors (consistent across all plots)
METHOD_ORDER = [
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
    "Koopman-RKDPmu",
    "PI-GRU",
]

METHOD_COLORS = {
    "IpDFT": "#1f77b4",  # blue
    "PLL": "#2ca02c",  # green
    "EKF": "#ff0000",  # red
    "EKF2": "#b22222",  # dark red
    "SOGI": "#ff00ff",  # magenta
    "RLS": "#00ced1",  # dark turquoise
    "Teager": "#ffa500",  # orange
    "TFT": "#800080",  # purple
    "RLS-VFF": "#008b8b",  # dark cyan
    "UKF": "#8b4513",  # saddle brown
    "Koopman-RKDPmu": "#808000",  # olive
    "PI-GRU": "#696969",  # dim gray
}


# ============================================================================
# 1. LOADING RAW RESULTS
# ============================================================================


def load_raw_results(
    root: str = RESULTS_RAW_DIR,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Walks through results_raw and builds:
        scenarios[scenario_name][method_name] = record (the JSON dict).

    Also returns the first encountered "global" metadata block as `global_cfg`.
    """
    scenarios: Dict[str, Dict[str, Any]] = defaultdict(dict)
    global_cfg: Dict[str, Any] = {}

    if not os.path.isdir(root):
        raise RuntimeError(f"results_raw directory not found: {root}")

    for scen_folder in sorted(os.listdir(root)):
        scen_path = os.path.join(root, scen_folder)
        if not os.path.isdir(scen_path):
            continue

        for fname in sorted(os.listdir(scen_path)):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(scen_path, fname)
            with open(fpath, "r") as f:
                rec = json.load(f)

            scen_meta = rec.get("metadata", {}).get("scenario", {})
            method_meta = rec.get("metadata", {}).get("method", {})

            scenario_name = scen_meta.get("name", scen_folder)
            method_name = method_meta.get("name", os.path.splitext(fname)[0])

            scenarios[scenario_name][method_name] = rec

            if not global_cfg:
                global_cfg = rec.get("metadata", {}).get("global", {})

    if not scenarios:
        raise RuntimeError("No raw JSON results found in results_raw/.")

    return scenarios, global_cfg


def available_scenarios_ordered(scenarios: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Returns scenario names in SCENARIO_ORDER first, then any extra ones sorted.
    """
    keys = set(scenarios.keys())
    ordered = [s for s in SCENARIO_ORDER if s in keys]
    remaining = sorted(list(keys - set(ordered)))
    ordered.extend(remaining)
    return ordered


def available_methods_ordered(scenarios: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Returns methods in METHOD_ORDER first, then any extra ones sorted.
    """
    methods = set()
    for scen_data in scenarios.values():
        methods.update(scen_data.keys())

    ordered = [m for m in METHOD_ORDER if m in methods]
    remaining = sorted(list(methods - set(ordered)))
    ordered.extend(remaining)
    return ordered


# ============================================================================
# 2. METRIC MATRICES & HELPERS
# ============================================================================


def build_metric_matrix(
    scenarios: Dict[str, Dict[str, Any]],
    metric_key: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Builds a matrix M (n_methods x n_scenarios) for a given metric.
    """
    scen_list = available_scenarios_ordered(scenarios)
    method_list = available_methods_ordered(scenarios)

    M = np.full((len(method_list), len(scen_list)), np.nan, dtype=float)

    for j, sname in enumerate(scen_list):
        for i, mname in enumerate(method_list):
            rec = scenarios[sname].get(mname)
            if rec is None:
                continue
            metrics = rec.get("metrics", {})
            if metric_key in metrics:
                try:
                    M[i, j] = float(metrics[metric_key])
                except Exception:
                    M[i, j] = np.nan

    return method_list, scen_list, M


def pretty_scen_labels(scen_list: List[str]) -> List[str]:
    return [SCENARIO_LABELS.get(s, s) for s in scen_list]


def safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x[x <= 0] = np.nan
    return np.log10(x)


# ============================================================================
# 3. MAIN FIGURES
# ============================================================================

# --------------------------------------------------------------------------
# 3.1. RMSE bar chart – single IEEE column
# --------------------------------------------------------------------------


def plot_rmse_bar(scenarios: Dict[str, Dict[str, Any]]) -> None:
    methods, scen_list, M = build_metric_matrix(scenarios, "RMSE")
    n_methods, n_scen = M.shape

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(n_scen)
    total_width = 0.8
    bar_width = total_width / n_methods

    for i, m in enumerate(methods):
        color = METHOD_COLORS.get(m, None)
        offset = (i - n_methods / 2) * bar_width + bar_width / 2
        vals = M[i, :]
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=m,
            color=color,
            edgecolor="black",
            linewidth=0.2,
        )

    ax.set_yscale("log")
    ax.set_ylabel("RMSE [Hz]")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_scen_labels(scen_list), rotation=20, ha="right")
    ax.set_xlabel("Scenario")

    # Log y-axis ticks more readable
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(1, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, axis="y", which="both")

    # Put legend outside, but keep single-column width
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.45, left=0.12, right=0.98, top=0.96)

    out_path = os.path.join(FIG_DIR, "SUMMARY_RMSE_BAR.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3.2. Heatmaps: generic helper + RMSE + Risk
# --------------------------------------------------------------------------


def plot_heatmap(
    M: np.ndarray,
    methods: List[str],
    scen_list: List[str],
    title: str,
    cbar_label: str,
    fname: str,
    log_values: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    data = np.array(M, dtype=float)
    if log_values:
        data = safe_log10(data)

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="viridis",
    )

    ax.set_xticks(np.arange(len(scen_list)))
    ax.set_xticklabels(pretty_scen_labels(scen_list), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    fig.subplots_adjust(left=0.20, bottom=0.20, right=0.97, top=0.92)

    out_path = os.path.join(FIG_DIR, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_heatmap(scenarios: Dict[str, Dict[str, Any]]) -> None:
    """
    Heatmap de log10(RMSE) [Hz] ordenando los métodos por
    RMSE medio global (mejor arriba, peor abajo).
    """
    methods, scen_list, M = build_metric_matrix(scenarios, "RMSE")

    # RMSE medio por método (ignorando NaNs)
    mean_rmse = np.nanmean(M, axis=1)
    order = np.argsort(mean_rmse)  # menor → mejor

    methods_sorted = [methods[i] for i in order]
    M_sorted = M[order, :]

    plot_heatmap(
        M_sorted,
        methods_sorted,
        scen_list,
        title="RMSE Heatmap (Method vs Scenario)",
        cbar_label="log$_{10}$(RMSE [Hz])",
        fname="HEATMAP_RMSE.png",
        log_values=True,
    )


def plot_risk_heatmap(scenarios: Dict[str, Dict[str, Any]]) -> None:
    methods, scen_list, M = build_metric_matrix(scenarios, "TRIP_TIME_0p5")
    plot_heatmap(
        M,
        methods,
        scen_list,
        title="Risk Heatmap (Spurious Trip Time)",
        cbar_label="log$_{10}$(Total time |e|>0.5 Hz [s])",
        fname="HEATMAP_RISK.png",
        log_values=True,
    )


# --------------------------------------------------------------------------
# 3.3. Performance profiles (RMSE, Risk, Complexity)
# --------------------------------------------------------------------------


def build_perf_profile(
    M: np.ndarray,
    methods: List[str],
    tau_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Given a metric matrix M (n_methods x n_scenarios), builds
    performance profiles for each method.
    """
    profiles: Dict[str, np.ndarray] = {}
    # best per scenario
    best = np.nanmin(M, axis=0)  # shape (n_scen,)
    ratios = M / best  # larger = worse

    for i, m in enumerate(methods):
        r = ratios[i, :]
        valid = ~np.isnan(r)
        if not valid.any():
            continue
        r = r[valid]
        frac = np.array([(r <= tau).mean() for tau in tau_grid], dtype=float)
        profiles[m] = frac
    return profiles


def plot_perf_profile_generic(
    scenarios: Dict[str, Dict[str, Any]],
    metric_key: str,
    fname: str,
    xlabel: str,
    title: str,
    tau_min: float = 1.0,
    tau_max: float = 1e3,
) -> None:
    methods, _, M = build_metric_matrix(scenarios, metric_key)

    # Remove methods that have all NaNs
    mask_valid_method = ~np.all(np.isnan(M), axis=1)
    M = M[mask_valid_method, :]
    methods = [m for m, ok in zip(methods, mask_valid_method) if ok]

    tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max), 200)
    profiles = build_perf_profile(M, methods, tau_grid)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for m in methods:
        frac = profiles[m]
        color = METHOD_COLORS.get(m, None)
        ax.plot(tau_grid, frac, label=m, color=color)

    ax.set_xscale("log")
    ax.set_xlim(tau_min, tau_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of scenarios")
    ax.set_title(title)

    ax.grid(True, which="both", axis="both")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.45, left=0.15, right=0.98, top=0.95)

    out_path = os.path.join(FIG_DIR, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_perf_profile_rmse(scenarios: Dict[str, Dict[str, Any]]) -> None:
    plot_perf_profile_generic(
        scenarios,
        metric_key="RMSE",
        fname="PERF_PROFILE_RMSE.png",
        xlabel=r"$\tau$ (RMSE factor over best)",
        title="Performance Profile (RMSE)",
        tau_min=1.0,
        tau_max=1e3,
    )


def plot_perf_profile_risk(scenarios: Dict[str, Dict[str, Any]]) -> None:
    # Risk: time |e|>0.5; some methods may have 0 => avoid 0 in ratios
    methods, _, M = build_metric_matrix(scenarios, "TRIP_TIME_0p5")
    M[M <= 0] = np.nan  # 0 = perfect behaviour; ignored in ratios
    tau_grid = np.logspace(0, 6, 200)
    profiles = build_perf_profile(M, methods, tau_grid)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for m in methods:
        if m not in profiles:
            continue
        frac = profiles[m]
        color = METHOD_COLORS.get(m, None)
        ax.plot(tau_grid, frac, label=m, color=color)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$ (trip time factor)")
    ax.set_ylabel("Fraction of scenarios")
    ax.set_title("Performance Profile (Risk |e|>0.5 Hz)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.45, left=0.17, right=0.98, top=0.95)

    out_path = os.path.join(FIG_DIR, "PERF_PROFILE_RISK.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_perf_profile_complexity(scenarios: Dict[str, Dict[str, Any]]) -> None:
    plot_perf_profile_generic(
        scenarios,
        metric_key="TIME_PER_SAMPLE_US",
        fname="PERF_PROFILE_COMPLEXITY.png",
        xlabel=r"$\tau$ (time per sample factor)",
        title="Performance Profile (Complexity)",
        tau_min=1.0,
        tau_max=1e3,
    )


# --------------------------------------------------------------------------
# 3.4. Pareto accuracy–complexity per scenario
# --------------------------------------------------------------------------


def pareto_front_indices(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Simple Pareto front: points such that no other point is strictly better
    in both x and y (x,y small is better).
    Returns boolean mask of shape (N,).
    """
    n = len(xs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if j == i or np.isnan(xs[j]) or np.isnan(ys[j]):
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                mask[i] = False
                break
    return mask


def plot_pareto_per_scenario(scenarios: Dict[str, Dict[str, Any]]) -> None:
    scen_list = available_scenarios_ordered(scenarios)
    methods_all = available_methods_ordered(scenarios)

    for sname in scen_list:
        scen_data = scenarios[sname]

        xs = []  # time per sample
        ys = []  # RMSE
        labels = []

        for m in methods_all:
            rec = scen_data.get(m)
            if rec is None:
                continue
            metrics = rec.get("metrics", {})
            if ("TIME_PER_SAMPLE_US" not in metrics) or ("RMSE" not in metrics):
                continue
            xs.append(float(metrics["TIME_PER_SAMPLE_US"]))
            ys.append(float(metrics["RMSE"]))
            labels.append(m)

        if not xs:
            continue

        xs = np.array(xs)
        ys = np.array(ys)
        labels = np.array(labels)

        mask_pareto = pareto_front_indices(xs, ys)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        for i, m in enumerate(labels):
            color = METHOD_COLORS.get(m, None)
            ax.scatter(
                xs[i],
                ys[i],
                s=18 if mask_pareto[i] else 14,
                marker="o",
                edgecolors="black" if mask_pareto[i] else "none",
                linewidths=0.4,
                color=color,
                zorder=3 if mask_pareto[i] else 2,
            )

        # log axes
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Time per sample [µs]")
        ax.set_ylabel("RMSE [Hz]")
        ax.set_title(f"Pareto Trade-off (Accuracy vs Complexity)\n{sname}")

        ax.grid(True, which="both")

        # Legend: show method names, mark Pareto ones with "(Pareto)"
        handles = []
        labels_leg = []
        for i, m in enumerate(labels):
            color = METHOD_COLORS.get(m, None)
            marker_edge = "black" if mask_pareto[i] else "none"
            h = ax.scatter(
                [],
                [],
                s=18 if mask_pareto[i] else 14,
                marker="o",
                edgecolors=marker_edge,
                linewidths=0.4,
                color=color,
            )
            handles.append(h)
            lbl = m + (" (Pareto)" if mask_pareto[i] else "")
            labels_leg.append(lbl)

        ax.legend(
            handles,
            labels_leg,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )

        fig.subplots_adjust(left=0.18, bottom=0.18, right=0.78, top=0.86)

        out_name = f"{sname}_PARETO_RMSE_COMPLEXITY.png"
        out_path = os.path.join(FIG_DIR, out_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------
# 3.5. Radar plots (optional, for “personality” summaries)
# --------------------------------------------------------------------------


def _normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(values.values()), dtype=float)
    v_min = np.nanmin(vals)
    v_max = np.nanmax(vals)
    if not np.isfinite(v_min) or v_max == v_min:
        return {k: 0.8 for k in values.keys()}  # all equal
    scores = {}
    for k, v in values.items():
        scores[k] = (v_max - v) / (v_max - v_min)
    return scores


def build_radar_scores(
    scenarios: Dict[str, Dict[str, Any]],
    methods_subset: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Returns normalized scores in [0,1] for each method in methods_subset:
        Accuracy   ~ mean RMSE (lower -> better)
        Speed      ~ mean TIME_PER_SAMPLE_US (lower -> better)
        Latency    ~ mean STRUCTURAL_LATENCY_MS (lower -> better)
        TripSafety ~ total TRIP_TIME_0p5 (lower -> better)
        BurstSafety~ total MAX_CONTIGUOUS_0p5 (lower -> better)
    """
    metrics_acc = {}
    metrics_speed = {}
    metrics_lat = {}
    metrics_trip = {}
    metrics_burst = {}

    for m in methods_subset:
        rmse_vals = []
        tps_vals = []
        lat_vals = []
        trip_vals = []
        burst_vals = []

        for scen_data in scenarios.values():
            rec = scen_data.get(m)
            if rec is None:
                continue
            met = rec.get("metrics", {})
            if "RMSE" in met:
                rmse_vals.append(float(met["RMSE"]))
            if "TIME_PER_SAMPLE_US" in met:
                tps_vals.append(float(met["TIME_PER_SAMPLE_US"]))
            if "STRUCTURAL_LATENCY_MS" in met:
                lat_vals.append(float(met["STRUCTURAL_LATENCY_MS"]))
            if "TRIP_TIME_0p5" in met:
                trip_vals.append(float(met["TRIP_TIME_0p5"]))
            if "MAX_CONTIGUOUS_0p5" in met:
                burst_vals.append(float(met["MAX_CONTIGUOUS_0p5"]))

        if not rmse_vals:
            continue

        metrics_acc[m] = np.mean(rmse_vals)
        metrics_speed[m] = np.mean(tps_vals) if tps_vals else np.nan
        metrics_lat[m] = np.mean(lat_vals) if lat_vals else np.nan
        metrics_trip[m] = np.sum(trip_vals) if trip_vals else np.nan
        metrics_burst[m] = np.sum(burst_vals) if burst_vals else np.nan

    # Normalizar (menor es mejor)
    acc = _normalize_scores(metrics_acc)
    spd = _normalize_scores(metrics_speed)
    lat = _normalize_scores(metrics_lat)
    trip = _normalize_scores(metrics_trip)
    burst = _normalize_scores(metrics_burst)

    scores: Dict[str, Dict[str, float]] = {}
    for m in methods_subset:
        if m not in acc:
            continue
        scores[m] = {
            "Accuracy": acc[m],
            "Speed": spd[m],
            "Latency": lat[m],
            "TripSafety": trip[m],
            "BurstSafety": burst[m],
        }

    return scores


def plot_radar_group(
    scenarios: Dict[str, Dict[str, Any]],
    methods_subset: List[str],
    title: str,
    fname: str,
) -> None:
    scores = build_radar_scores(scenarios, methods_subset)
    if not scores:
        return

    metrics_labels = ["Accuracy", "Speed", "Latency", "TripSafety", "BurstSafety"]
    n_metrics = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig = plt.figure(figsize=(2.8, 2.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # radial limits
    ax.set_rlabel_position(0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=7)

    for m in methods_subset:
        if m not in scores:
            continue
        vals = [scores[m][lab] for lab in metrics_labels]
        vals += vals[:1]
        color = METHOD_COLORS.get(m, None)
        ax.plot(angles, vals, linewidth=1.0, label=m, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), frameon=False)

    fig.subplots_adjust(left=0.10, right=0.80, top=0.90, bottom=0.05)

    out_path = os.path.join(FIG_DIR, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_radars(scenarios: Dict[str, Dict[str, Any]]) -> None:
    # Puedes ajustar los grupos según tu paper
    classics = ["IpDFT", "PLL", "RLS", "SOGI", "Teager"]
    advanced = ["EKF", "EKF2", "TFT", "RLS-VFF", "UKF"]
    futuristic = ["Koopman-RKDPmu", "PI-GRU"]

    plot_radar_group(
        scenarios,
        classics,
        title="Personality Radar – Classics",
        fname="RADAR_CLASSICS.png",
    )
    plot_radar_group(
        scenarios,
        advanced,
        title="Personality Radar – Advanced",
        fname="RADAR_ADVANCED.png",
    )
    plot_radar_group(
        scenarios,
        futuristic,
        title="Personality Radar – Futuristic",
        fname="RADAR_FUTURISTIC.png",
    )


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    scenarios, global_cfg = load_raw_results()
    print("Loaded scenarios:", ", ".join(sorted(scenarios.keys())))
    print("Global config:", global_cfg)

    # --- Core summary plots ---
    plot_rmse_bar(scenarios)
    plot_rmse_heatmap(scenarios)
    plot_risk_heatmap(scenarios)

    plot_perf_profile_rmse(scenarios)
    plot_perf_profile_risk(scenarios)
    plot_perf_profile_complexity(scenarios)

    plot_pareto_per_scenario(scenarios)

    # Optional radar plots (comment out if not needed in the paper)
    plot_all_radars(scenarios)

    print(f"All figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
