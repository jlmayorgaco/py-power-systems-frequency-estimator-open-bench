import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================================================
# Configuración Estilo IEEE para Journal Q1
# ============================================================
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "figure.autolayout": True,
    }
)


def plot_comparison(
    scenario_id, base_path="artifacts/waveforms", output_dir="artifacts/figures"
):
    """
    Genera una comparativa completa de la respuesta dinámica.
    Exporta en PDF (vectorial para el paper) y PNG (raster para revisión rápida).
    """
    csv_path = os.path.join(base_path, scenario_id, "all.csv")
    if not os.path.exists(csv_path):
        print(f"  [!] Saltando: No se encontró all.csv en {scenario_id}")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # 1. Referencia de Frecuencia Real
    ax.plot(df["t"], df["real_f"], "k--", label="True Frequency", alpha=0.8, lw=1.5)

    # 2. Graficar cada método estimado
    # Excluimos columnas de tiempo, referencia y voltaje
    methods = [c for c in df.columns if c not in ["t", "real_f", "real_v"]]
    for m in methods:
        label_clean = m.replace("_", "-")
        ax.plot(df["t"], df[m], label=label_clean, lw=1.3)

    ax.set_title(f"Dynamic Response: {scenario_id.replace('_', ' ')}")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best", fontsize=9, frameon=True)

    # --- CREACIÓN DE DIRECTORIOS Y GUARDADO DUAL ---
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.join(output_dir, f"{scenario_id}_full")

    # Guardar PDF (300 DPI)
    plt.savefig(f"{base_filename}.pdf", bbox_inches="tight", dpi=300)
    # Guardar PNG (300 DPI) para visualización rápida
    plt.savefig(f"{base_filename}.png", bbox_inches="tight", dpi=300)

    plt.close()
    print(f"  [OK] Full plot guardado (PDF+PNG): {base_filename}")


def plot_nadir_zoom(
    scenario_id, base_path="artifacts/waveforms", output_dir="artifacts/figures"
):
    """
    Zoom automático al transitorio (Nadir) para evidenciar latencia.
    Crucial para el argumento contra métodos de ventana como IpDFT.
    """
    csv_path = os.path.join(base_path, scenario_id, "all.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    # --- LÓGICA DE AUTO-ZOOM AL TRANSITORIO ---
    # Detectamos el punto crítico (mínimo de frecuencia real)
    idx_nadir = df["real_f"].idxmin()
    t_nadir = df["t"].iloc[idx_nadir]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Graficar con mayor grosor para el zoom de auditoría
    ax.plot(df["t"], df["real_f"], "k--", label="True", lw=2, zorder=5)

    methods = [c for c in df.columns if c not in ["t", "real_f", "real_v"]]
    for m in methods:
        label_clean = m.replace("_", "-")
        ax.plot(df["t"], df[m], label=label_clean, lw=1.5)

    # Ventana de Zoom: Foco en la caída y recuperación (100ms antes, 500ms después)
    ax.set_xlim([t_nadir - 0.1, t_nadir + 0.5])

    # Ajuste dinámico del eje Y para resaltar la desviación
    f_min = df["real_f"].min()
    ax.set_ylim([f_min - 0.05, 60.05])

    ax.set_title(f"Transient Transition Zoom (Nadir)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, loc="upper right", frameon=True)

    # --- GUARDADO DUAL ---
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.join(output_dir, f"{scenario_id}_zoom")

    plt.savefig(f"{base_filename}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_filename}.png", bbox_inches="tight", dpi=300)

    plt.close()
    print(f"  [OK] Nadir zoom guardado (PDF+PNG): {base_filename}")
