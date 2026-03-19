#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extra_plots_from_json.py

Genera figuras adicionales a partir de benchmark_results.json:

1) Heatmap de RMSE (método x escenario, reordenado)
2) Heatmap de TRIP_TIME_0p5
3) Performance profiles (Dolan–Moré) para:
   - RMSE
   - TIME_PER_SAMPLE_US
   - TRIP_TIME_0p5
4) Radar "personalidad Pokémon" para grupos de métodos
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# 0. CONFIGURACIÓN BÁSICA (NO BORRAMOS NADA)
# =============================================================

OUTPUT_DIR = "figures_estimatores_benchmark_test5"
JSON_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo IEEE similar al de plotting.py, pero sin borrar carpeta
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.figsize": (3.5, 2.8),
        "lines.linewidth": 1.0,
        "grid.alpha": 0.4,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.autolayout": True,
    }
)

# Paleta de colores coherente con plotting.py
DEFAULT_COLORS = {
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
    "MLFreqNet": "black",
    "PI-GRU": "black",  # nombre real en el JSON
}


# =============================================================
# 1. UTILIDADES
# =============================================================


def load_json(path=JSON_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el JSON en {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_scenarios_methods(json_data):
    scenarios = list(json_data["results"].keys())
    # Asumimos que todos tienen los mismos métodos
    first_sc = scenarios[0]
    methods = list(json_data["results"][first_sc]["methods"].keys())
    return scenarios, methods


def get_metric_matrix(json_data, metric_key):
    """
    Devuelve matriz [n_methods x n_scenarios] del metric_key.
    También devuelve (scenarios, methods).
    """
    scenarios, methods = get_scenarios_methods(json_data)
    M = np.zeros((len(methods), len(scenarios)))
    for j, sc in enumerate(scenarios):
        for i, m in enumerate(methods):
            M[i, j] = json_data["results"][sc]["methods"][m][metric_key]
    return M, scenarios, methods


# =============================================================
# 2. HEATMAPS (RMSE y RIESGO)
# =============================================================


def save_heatmap_rmse(json_data):
    """
    Heatmap de RMSE:
    - Columnas reordenadas: Mag_Step, Modulation, Freq_Ramp, Nightmare, MultiEvent
    - Filas ordenadas automáticamente por RMSE medio (mejores arriba),
      forzando a Teager al fondo.
    """
    results = json_data["results"]

    # --- 1) Orden deseado de escenarios (swap Ramp <-> Modulation) ---
    desired_order = [
        "IEEE_Mag_Step",
        "IEEE_Modulation",
        "IEEE_Freq_Ramp",
        "IBR_Nightmare",
        "IBR_MultiEvent",
    ]
    # Nos quedamos solo con los que existan en el JSON
    scenarios = [sc for sc in desired_order if sc in results]

    # --- 2) Lista de métodos (asumimos que todos los escenarios tienen los mismos) ---
    first_sc = scenarios[0]
    methods_all = list(results[first_sc]["methods"].keys())

    # --- 3) Construir matriz RMSE (methods x scenarios) ---
    rmse_mat = np.zeros((len(methods_all), len(scenarios)))
    for i, m in enumerate(methods_all):
        for j, sc in enumerate(scenarios):
            rmse_mat[i, j] = results[sc]["methods"][m]["RMSE"]

    # --- 4) Ordenar métodos por RMSE medio (log10), mejores arriba ---
    log_rmse = np.log10(rmse_mat + 1e-12)
    mean_log_rmse = log_rmse.mean(axis=1)
    sort_idx = np.argsort(mean_log_rmse)  # menor → mejor

    ordered_methods = [methods_all[i] for i in sort_idx]

    # Forzar Teager al fondo (si existe)
    if "Teager" in ordered_methods:
        ordered_methods.remove("Teager")
        ordered_methods.append("Teager")

    # Re-construir la matriz en el nuevo orden de métodos
    rmse_mat_ord = np.zeros_like(rmse_mat)
    for i, m in enumerate(ordered_methods):
        m_idx = methods_all.index(m)
        rmse_mat_ord[i, :] = rmse_mat[m_idx, :]

    log_rmse_ord = np.log10(rmse_mat_ord + 1e-12)

    # --- 5) Etiquetas bonitas para escenarios ---
    scenario_labels_map = {
        "IEEE_Mag_Step": "Mag Step",
        "IEEE_Freq_Ramp": "Freq Ramp",
        "IEEE_Modulation": "Modulation",
        "IBR_Nightmare": "Nightmare",
        "IBR_MultiEvent": "Multi-Event",
    }
    xlabels = [scenario_labels_map.get(sc, sc) for sc in scenarios]

    # --- 6) Plot ---
    fig, ax = plt.subplots(figsize=(7.16, 3.5))  # double-column IEEE

    im = ax.imshow(
        log_rmse_ord,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="viridis",
    )

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(np.arange(len(ordered_methods)))
    ax.set_yticklabels(ordered_methods)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.set_title("RMSE Heatmap (Method vs Scenario)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(RMSE [Hz])")

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "EXTRA_HEATMAP_RMSE.png"))
    plt.close()


def save_heatmap_triptime(json_data):
    M, scenarios, methods = get_metric_matrix(json_data, "TRIP_TIME_0p5")
    M_plot = np.log10(M + 1e-6)  # algunos pueden ser 0

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(M_plot, origin="lower", aspect="auto", cmap="magma")

    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=0)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(Total time |e|>0.5 Hz [s])")

    ax.set_title("Risk Heatmap (Spurious Trip Time)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "EXTRA_HEATMAP_TRIPTIME.png"))
    plt.close()


# =============================================================
# 3. PERFORMANCE PROFILES (DOLAN–MORÉ)
# =============================================================


def compute_performance_profile(json_data, metric_key):
    """
    Devuelve:
      taus: vector de thresholds
      profiles: dict[method] -> valores p_m(tau)
    """
    scenarios, methods = get_scenarios_methods(json_data)

    # M[s, m] = métrica (problema s, método m)
    S = len(scenarios)
    M = np.zeros((S, len(methods)))

    for s, sc in enumerate(scenarios):
        for i, m in enumerate(methods):
            val = json_data["results"][sc]["methods"][m][metric_key]
            # Evitar cero que rompe ratios
            if val <= 0:
                val = 1e-12
            M[s, i] = val

    # Ratios r_{s,m} = metric / best_metric_over_m
    ratios = np.zeros_like(M)
    for s in range(S):
        best = np.min(M[s, :])
        ratios[s, :] = M[s, :] / best

    # Rango de taus en log
    max_ratio = np.max(ratios)
    max_ratio = max(max_ratio, 2.0)
    taus = np.logspace(0, np.log10(max_ratio), 200)

    profiles = {}
    for i, m in enumerate(methods):
        r_m = ratios[:, i]
        # p_m(tau) = fracción de problemas con r_{s,m} <= tau
        p_vals = []
        for tau in taus:
            p = np.mean(r_m <= tau)
            p_vals.append(p)
        profiles[m] = np.array(p_vals)

    return taus, profiles


def save_performance_profiles(json_data):
    # ---- 1) RMSE ----
    taus, prof_rmse = compute_performance_profile(json_data, "RMSE")
    plt.figure(figsize=(3.5, 2.8))
    for m, p in prof_rmse.items():
        plt.plot(
            taus,
            p,
            label=m,
            linewidth=1.0,
            alpha=0.9,
            color=DEFAULT_COLORS.get(m, None),
        )
    plt.xscale("log")
    plt.xlabel(r"$\tau$ (RMSE factor over best)")
    plt.ylabel("Fraction of scenarios")
    plt.title("Performance Profile (RMSE)")
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=6, ncol=2, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "EXTRA_PERF_PROFILE_RMSE.png"))
    plt.close()

    # ---- 2) TIME_PER_SAMPLE_US ----
    taus_t, prof_time = compute_performance_profile(json_data, "TIME_PER_SAMPLE_US")
    plt.figure(figsize=(3.5, 2.8))
    for m, p in prof_time.items():
        plt.plot(
            taus_t,
            p,
            label=m,
            linewidth=1.0,
            alpha=0.9,
            color=DEFAULT_COLORS.get(m, None),
        )
    plt.xscale("log")
    plt.xlabel(r"$\tau$ (time per sample factor)")
    plt.ylabel("Fraction of scenarios")
    plt.title("Performance Profile (Complexity)")
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=6, ncol=2, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "EXTRA_PERF_PROFILE_TIME.png"))
    plt.close()

    # ---- 3) TRIP_TIME_0p5 (riesgo) ----
    taus_r, prof_risk = compute_performance_profile(json_data, "TRIP_TIME_0p5")
    plt.figure(figsize=(3.5, 2.8))
    for m, p in prof_risk.items():
        plt.plot(
            taus_r,
            p,
            label=m,
            linewidth=1.0,
            alpha=0.9,
            color=DEFAULT_COLORS.get(m, None),
        )
    plt.xscale("log")
    plt.xlabel(r"$\tau$ (trip time factor)")
    plt.ylabel("Fraction of scenarios")
    plt.title("Performance Profile (Risk |e|>0.5 Hz)")
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=6, ncol=2, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "EXTRA_PERF_PROFILE_RISK.png"))
    plt.close()


# =============================================================
# 4. RADAR "POKÉMON PERSONALITY"
# =============================================================


def compute_personality_scores(json_data):
    """
    Para cada método, calcula promedios por escenario de:
      - RMSE
      - TIME_PER_SAMPLE_US
      - TRIP_TIME_0p5
      - MAX_CONTIGUOUS_0p5
      - STRUCTURAL_LATENCY_MS

    Luego los normaliza a [0, 1] donde 1 es "mejor".
    Devuelve:
      metrics_norm[metric_name][method] = score in [0,1]
    """
    scenarios, methods = get_scenarios_methods(json_data)

    raw = {
        m: {
            "RMSE": [],
            "TIME_PER_SAMPLE_US": [],
            "TRIP_TIME_0p5": [],
            "MAX_CONTIGUOUS_0p5": [],
            "STRUCTURAL_LATENCY_MS": [],
        }
        for m in methods
    }

    for sc in scenarios:
        for m in methods:
            vals = json_data["results"][sc]["methods"][m]
            raw[m]["RMSE"].append(vals["RMSE"])
            raw[m]["TIME_PER_SAMPLE_US"].append(vals["TIME_PER_SAMPLE_US"])
            raw[m]["TRIP_TIME_0p5"].append(vals["TRIP_TIME_0p5"])
            raw[m]["MAX_CONTIGUOUS_0p5"].append(vals["MAX_CONTIGUOUS_0p5"])
            raw[m]["STRUCTURAL_LATENCY_MS"].append(vals["STRUCTURAL_LATENCY_MS"])

    # Promedios por método
    avg = {m: {} for m in methods}
    for m in methods:
        for k in raw[m]:
            arr = np.array(raw[m][k], dtype=float)
            avg[m][k] = float(np.mean(arr))

    # Normalización: todas estas métricas "más pequeño es mejor"
    # score = (max - val)/(max - min + eps)
    metrics_norm = {
        k: {} for k in ["Accuracy", "Speed", "TripSafety", "BurstSafety", "Latency"]
    }

    # Helpers para normalizar
    def normalize_inverted(values_dict):
        vals = np.array(list(values_dict.values()), dtype=float)
        vmin = np.min(vals)
        vmax = np.max(vals)
        eps = 1e-12
        scores = {}
        for m, v in values_dict.items():
            scores[m] = float((vmax - v) / (vmax - vmin + eps))
        return scores

    # Accuracy: basado en RMSE
    rmse_vals = {m: avg[m]["RMSE"] for m in methods}
    metrics_norm["Accuracy"] = normalize_inverted(rmse_vals)

    # Speed: basado en TIME_PER_SAMPLE_US
    time_vals = {m: avg[m]["TIME_PER_SAMPLE_US"] for m in methods}
    metrics_norm["Speed"] = normalize_inverted(time_vals)

    # TripSafety: TRIP_TIME_0p5
    trip_vals = {m: avg[m]["TRIP_TIME_0p5"] for m in methods}
    metrics_norm["TripSafety"] = normalize_inverted(trip_vals)

    # BurstSafety: MAX_CONTIGUOUS_0p5
    burst_vals = {m: avg[m]["MAX_CONTIGUOUS_0p5"] for m in methods}
    metrics_norm["BurstSafety"] = normalize_inverted(burst_vals)

    # Latency: STRUCTURAL_LATENCY_MS
    lat_vals = {m: avg[m]["STRUCTURAL_LATENCY_MS"] for m in methods}
    metrics_norm["Latency"] = normalize_inverted(lat_vals)

    return metrics_norm, methods


def radar_plot_group(metrics_norm, methods_group, title, filename):
    """
    Dibuja un radar chart para un grupo de métodos.

    Ejes:
      - Accuracy
      - Speed
      - TripSafety
      - BurstSafety
      - Latency
    """
    metrics_names = ["Accuracy", "Speed", "TripSafety", "BurstSafety", "Latency"]
    N = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # cerrar polígono

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, polar=True)

    for m in methods_group:
        # Saltar métodos que no existan en el JSON
        if m not in metrics_norm["Accuracy"]:
            continue
        values = [metrics_norm[k][m] for k in metrics_names]
        values = np.concatenate([values, [values[0]]])
        ax.plot(
            angles,
            values,
            label=m,
            linewidth=1.0,
            alpha=0.9,
            color=DEFAULT_COLORS.get(m, None),
        )
        ax.fill(angles, values, alpha=0.15, color=DEFAULT_COLORS.get(m, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
    ax.set_ylim(0, 1.0)

    ax.set_title(title, va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def save_radar_personalities(json_data):
    metrics_norm, all_methods = compute_personality_scores(json_data)

    # Grupos "tipo Pokémon"
    basic_group = [
        m for m in ["IpDFT", "PLL", "RLS", "SOGI", "Teager"] if m in all_methods
    ]

    advanced_group = [
        m for m in ["EKF", "EKF2", "TFT", "RLS-VFF", "UKF"] if m in all_methods
    ]

    future_group = [m for m in ["Koopman-RKDPmu", "PI-GRU"] if m in all_methods]

    if basic_group:
        radar_plot_group(
            metrics_norm,
            basic_group,
            "Personality Radar – Classics",
            "EXTRA_RADAR_CLASSICS.png",
        )

    if advanced_group:
        radar_plot_group(
            metrics_norm,
            advanced_group,
            "Personality Radar – Advanced",
            "EXTRA_RADAR_ADVANCED.png",
        )

    if future_group:
        radar_plot_group(
            metrics_norm,
            future_group,
            "Personality Radar – Futuristic",
            "EXTRA_RADAR_FUTURISTIC.png",
        )


# =============================================================
# 5. MAIN
# =============================================================


def main():
    print(f"Loading JSON from: {JSON_PATH}")
    data = load_json(JSON_PATH)

    print("Generating extra heatmaps...")
    save_heatmap_rmse(data)
    save_heatmap_triptime(data)

    print("Generating performance profiles...")
    save_performance_profiles(data)

    print("Generating radar 'Pokémon personalities'...")
    save_radar_personalities(data)

    print(f"Done. Extra plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
