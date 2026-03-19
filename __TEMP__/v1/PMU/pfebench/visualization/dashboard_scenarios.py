#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualization/dashboard_scenarios.py

Generador de Dashboards de Física para IEEE Transactions (JOURNAL READY).
Ejecuta directamente las clases de escenario, obtiene datos frescos y genera
gráficos comparativos.

Modos de Salida:
  1. Detallado (Pagina 1 y Pagina 2).
  2. Compacto (Todos los escenarios en una sola página).

Salida:
  - artifacts/figures/dashboard_physics_page_1.[pdf/png/svg]
  - artifacts/figures/dashboard_physics_page_2.[pdf/png/svg]
  - artifacts/figures/dashboard_physics_compact.[pdf/png/svg]
"""

import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import List, Dict, Any, Optional

# --- 1. CONFIGURACIÓN DE ESTILO IEEE ---
OUTPUT_DIR = "artifacts/figures"
STYLE_CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "lines.linewidth": 0.75,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.figsize": (8.5, 11),
    "savefig.dpi": 600,
    "figure.constrained_layout.use": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
}

COLORS = {
    "volt": "#0072B2",  # Azul IEEE
    "freq": "#D55E00",  # Bermellón/Naranja
    "ref": "#444444",  # Gris oscuro
    "hist": "#555555",  # Gris histograma
}

# --- 2. REGISTRO DE ESCENARIOS ---
SCENARIO_REGISTRY = {
    # G1
    "G1_E1_Pure_60Hz": "pfebench.scenarios.G1_E1_Pure_60Hz.G1_E1_Pure_60Hz",
    "G1_E2_Gaussian_Noise_1pct": "pfebench.scenarios.G1_E2_Gaussian_Noise_1pct.G1_E2_Gaussian_Noise_1pct",
    "G1_E3_Gaussian_Noise_5pct": "pfebench.scenarios.G1_E3_Gaussian_Noise_5pct.G1_E3_Gaussian_Noise_5pct",
    # G2
    "G2_E4_Voltage_Mag_Step_1pct": "pfebench.scenarios.G2_E4_Voltage_Mag_Step_1pct.G2_E4_Voltage_Mag_Step_1pct",
    "G2_E5_Voltage_Mag_Step_10pct": "pfebench.scenarios.G2_E5_Voltage_Mag_Step_10pct.G2_E5_Voltage_Mag_Step_10pct",
    "G2_E6_Freq_Step_60_to_59p5": "pfebench.scenarios.G2_E6_Freq_Step_60_to_59p5.G2_E6_Freq_Step_60_to_59p5",
    "G2_E7_Freq_Step_60_to_55": "pfebench.scenarios.G2_E7_Freq_Step_60_to_55.G2_E7_Freq_Step_60_to_55",
    "G2_E8_Fast_Ramp_plus5Hzs": "pfebench.scenarios.G2_E8_Fast_Ramp_plus5Hzs.G2_E8_Fast_Ramp_plus5Hzs",
    "G2_E9_Slow_Ramp_minus0p5Hzs": "pfebench.scenarios.G2_E9_Slow_Ramp_minus0p5Hzs.G2_E9_Slow_Ramp_minus0p5Hzs",
    # G3
    "G3_E10_AM_Modulation": "pfebench.scenarios.G3_E10_AM_Modulation.G3_E10_AM_Modulation",
    "G3_E11_FM_Modulation": "pfebench.scenarios.G3_E11_FM_Modulation.G3_E11_FM_Modulation",
    "G3_E12_Phase_Jump": "pfebench.scenarios.G3_E12_Phase_Jump.G3_E12_Phase_Jump",
    "G3_E13_Impulsive_Outliers": "pfebench.scenarios.G3_E13_Impulsive_Outliers.G3_E13_Impulsive_Outliers",
    "G3_E14_Noise_Harmonics": "pfebench.scenarios.G3_E14_Noise_Harmonics.G3_E14_Noise_Harmonics",
    "G3_E15_Noise_Interharmonics": "pfebench.scenarios.G3_E15_Noise_Interharmonics.G3_E15_Noise_Interharmonics",
    # G4
    "G4_E16_Composite_Islanding": "pfebench.scenarios.G4_E16_Composite_Islanding.G4_E16_Composite_Islanding",
    "G4_E17_Multi_Event_Profile": "pfebench.scenarios.G4_E17_Multi_Event_Profile.G4_E17_Multi_Event_Profile",
    "G4_E18_Chamorro_Event": "pfebench.scenarios.G4_E18_Chamorro_Event.G4_E18_Chamorro_Event",
}

# --- 3. CONFIGURACIÓN DE FILAS ---
PAGES = [
    # PAGINA 1
    [
        {
            "title": "G1: Gaussian Noise",
            "ids": [
                "G1_E3_Gaussian_Noise_5pct",
                "G1_E2_Gaussian_Noise_1pct",
                "G1_E1_Pure_60Hz",
            ],
            "labels": ["5%", "1%", "0%"],
            "xlim": (1.98, 2.02),
            "type": "multi",
            "hist_id": "G1_E3_Gaussian_Noise_5pct",
        },
        {
            "title": "G2: Voltage Steps",
            "ids": ["G2_E5_Voltage_Mag_Step_10pct", "G2_E4_Voltage_Mag_Step_1pct"],
            "labels": ["-10%", "-1%"],
            "xlim": (2.45, 2.55),
            "type": "multi",
        },
        {
            "title": "G2: Frequency Steps",
            "ids": ["G2_E7_Freq_Step_60_to_55", "G2_E6_Freq_Step_60_to_59p5"],
            "labels": ["-5Hz", "-0.5Hz"],
            "xlim": (2.45, 2.55),
            "type": "multi",
        },
        {
            "title": "G2: Frequency Ramps",
            "ids": ["G2_E8_Fast_Ramp_plus5Hzs", "G2_E9_Slow_Ramp_minus0p5Hzs"],
            "labels": ["+5Hz/s", "-0.5Hz/s"],
            "xlim": (0.8, 2.2),
            "type": "multi",
        },
        {
            "title": "G3-E10: AM Modulation",
            "ids": ["G3_E10_AM_Modulation"],
            "xlim": (0.0, 1.0),
            "type": "single",
        },
        {
            "title": "G3-E11: FM Modulation",
            "ids": ["G3_E11_FM_Modulation"],
            "xlim": (0.0, 1.0),
            "type": "single",
        },
        {
            "title": "G3-E12: Phase Jump",
            "ids": ["G3_E12_Phase_Jump"],
            "xlim": (2.45, 2.55),
            "type": "single",
            "annotation": {"type": "phase", "t": 2.5},
        },
    ],
    # PAGINA 2
    [
        {
            "title": "G3-E13: Impulsive Outliers",
            "ids": ["G3_E13_Impulsive_Outliers"],
            "xlim": (0.0, 0.5),
            "type": "single",
            "hist_id": "G3_E13_Impulsive_Outliers",
        },
        {
            "title": "G3-E14: Harmonics + Noise",
            "ids": ["G3_E14_Noise_Harmonics"],
            "xlim": (0.0, 0.08),
            "type": "single",
            "hist_id": "G3_E14_Noise_Harmonics",
            "hist_pos": "lower right",
        },
        {
            "title": "G3-E15: Interharmonics",
            "ids": ["G3_E15_Noise_Interharmonics"],
            "xlim": (0.0, 0.15),
            "type": "single",
            "hist_id": "G3_E15_Noise_Interharmonics",
            "hist_pos": "lower right",
        },
        {
            "title": "G4-E16: Composite Islanding",
            "ids": ["G4_E16_Composite_Islanding"],
            "xlim": (1.90, 2.20),
            "ylim_f": (59.5, 60.5),
            "type": "single",
            "annotation": {"type": "island_detail", "t": 2.0},
        },
        {
            "title": "G4-E17: Multi-Event Profile",
            "ids": ["G4_E17_Multi_Event_Profile"],
            "xlim": (0.8, 4.0),
            "type": "single",
            "hist_id": "G4_E17_Multi_Event_Profile",
            "hist_pos": "lower left",
            "annotation": {"type": "boss", "t1": 1.0, "t2": 2.5},
        },
        {
            "title": "G4-E18: Chamorro Event",
            "ids": ["G4_E18_Chamorro_Event"],
            "xlim": (0.0, 1.0),
            "type": "single",
        },
    ],
]


class DashboardGenerator:
    def __init__(self):
        self._setup_style()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._cache_results = {}

    def _setup_style(self):
        plt.rcParams.update(STYLE_CONFIG)

    def _load_scenario(self, sc_id: str):
        if sc_id in self._cache_results:
            return self._cache_results[sc_id]
        if sc_id not in SCENARIO_REGISTRY:
            return None

        full_path = SCENARIO_REGISTRY[sc_id]
        module_name, class_name = full_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            try:
                sc = cls()
                res = sc.run()
                self._cache_results[sc_id] = res
                return res
            except FileNotFoundError:
                return None
            except Exception as e:
                print(f"[ERROR] {sc_id}: {e}")
                return None
        except ImportError:
            return None

    def _plot_voltage(self, ax, res, label=None, ls="-"):
        max_points = 20000
        step = max(1, len(res.t) // max_points)
        t, v = res.t[::step], res.v[::step]
        ax.plot(
            t,
            v,
            color=COLORS["volt"],
            lw=STYLE_CONFIG["lines.linewidth"],
            label=label,
            linestyle=ls,
        )
        ax.set_ylabel("v(t) [p.u.]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(-1.2, 1.2)

    def _plot_frequency(self, ax, res, label=None, ls="-"):
        max_points = 20000
        step = max(1, len(res.t) // max_points)
        t, f = res.t[::step], res.f_true[::step]

        f_span = np.max(f) - np.min(f)
        if f_span < 0.05:
            ax.plot(
                t,
                f - 60.0,
                color=COLORS["freq"],
                lw=STYLE_CONFIG["lines.linewidth"],
                label=label,
                linestyle=ls,
            )
            ax.set_ylabel(r"$\Delta f$ [Hz]")
            ax.axhline(0, color=COLORS["ref"], ls=":", lw=0.5)
            ylim = 0.02
            if np.max(np.abs(f - 60.0)) > ylim:
                ylim = np.max(np.abs(f - 60.0)) * 1.1
            ax.set_ylim(-ylim, ylim)
        else:
            ax.plot(
                t,
                f,
                color=COLORS["freq"],
                lw=STYLE_CONFIG["lines.linewidth"],
                label=label,
                linestyle=ls,
            )
            ax.set_ylabel("f(t) [Hz]")
        ax.set_xlabel("Time [s]")

    def _add_noise_histogram(self, ax, sc_id, position="upper right"):
        res = self._cache_results.get(sc_id)
        if not res or res.state is None:
            return
        try:
            v_ideal = res.state.A * np.sin(res.state.phi)
            residual = res.state.v - v_ideal
            ax_ins = inset_axes(
                ax, width="35%", height="40%", loc=position, borderpad=1
            )
            ax_ins.hist(
                residual,
                bins=30,
                density=True,
                color=COLORS["hist"],
                alpha=0.75,
                edgecolor="none",
            )
            ax_ins.set_facecolor("white")
            ax_ins.patch.set_alpha(1.0)
            for spine in ax_ins.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.6)
            ax_ins.set_title("Noise Dist.", fontsize=5, pad=2)
            ax_ins.tick_params(
                labelleft=False, labelbottom=False, left=False, bottom=False
            )
            ax_ins.set_xlabel(r"$\epsilon$", fontsize=5, labelpad=1)
        except:
            pass

    def _annotate(self, ax_v, ax_f, conf):
        atype = conf.get("type")
        if atype == "phase":
            t = conf["t"]
            for ax in [ax_v, ax_f]:
                ax.axvline(t, color="crimson", ls="--", alpha=0.5, lw=0.8)
            ax_v.text(
                t,
                1.05,
                r"$\Delta \phi$",
                ha="center",
                va="bottom",
                color="crimson",
                fontsize=7,
                transform=ax_v.get_xaxis_transform(),
            )
        elif atype == "island_detail":
            t = conf["t"]
            ax_v.annotate(
                "Phase/V-Step",
                xy=(t, 0.7),
                xytext=(t - 0.08, 0.9),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
                fontsize=6,
                color="crimson",
                ha="right",
            )
            ax_f.axvline(t, color="crimson", ls="--", alpha=0.5, lw=0.8)
            ax_f.text(
                t + 0.01,
                60.05,
                "ROCOF",
                ha="left",
                va="bottom",
                color="crimson",
                fontsize=6,
            )
        elif atype == "boss":
            t1, t2 = conf["t1"], conf["t2"]
            for ax in [ax_f]:
                ax.axvline(t1, color="gray", ls=":", alpha=0.5)
                ax.axvline(t2, color="gray", ls=":", alpha=0.5)
                yl = ax.get_ylim()
                ax.text(
                    t1,
                    (yl[0] + yl[1]) / 2,
                    " Ramp",
                    ha="left",
                    va="center",
                    rotation=90,
                    color="gray",
                    fontsize=6,
                )
                ax.text(
                    t2,
                    (yl[0] + yl[1]) / 2,
                    " Jump",
                    ha="left",
                    va="center",
                    rotation=90,
                    color="gray",
                    fontsize=6,
                )

    def _render_row(self, ax_v, ax_f, row_conf):
        """Lógica común de renderizado para una fila."""
        ax_v.set_title(row_conf["title"], loc="left", fontsize=9, fontweight="medium")
        ids, labels = row_conf["ids"], row_conf.get(
            "labels", [None] * len(row_conf["ids"])
        )
        linestyles = ["-", "--", ":", "-."]
        has_data = False

        for j, sc_id in enumerate(ids):
            res = self._load_scenario(sc_id)
            if res:
                has_data = True
                lbl = labels[j]
                ls = linestyles[j % len(linestyles)]
                self._plot_voltage(ax_v, res, label=lbl, ls=ls)
                self._plot_frequency(ax_f, res, label=lbl, ls=ls)

        if has_data:
            xlim = row_conf.get("xlim")
            if "xlim_v" in row_conf:
                ax_v.set_xlim(row_conf["xlim_v"])
            elif xlim:
                ax_v.set_xlim(xlim)

            if "xlim_f" in row_conf:
                ax_f.set_xlim(row_conf["xlim_f"])
            elif xlim:
                ax_f.set_xlim(xlim)

            if "ylim_v" in row_conf:
                ax_v.set_ylim(row_conf["ylim_v"])
            if "ylim_f" in row_conf:
                ax_f.set_ylim(row_conf["ylim_f"])

            if "annotation" in row_conf:
                self._annotate(ax_v, ax_f, row_conf["annotation"])
            if "hist_id" in row_conf:
                self._add_noise_histogram(
                    ax_f,
                    row_conf["hist_id"],
                    position=row_conf.get("hist_pos", "upper right"),
                )
            if row_conf["type"] == "multi":
                ax_v.legend(
                    fontsize=6,
                    loc="upper right",
                    frameon=True,
                    framealpha=0.9,
                    fancybox=False,
                )
        else:
            ax_v.text(
                0.5,
                0.5,
                "Data Error",
                ha="center",
                va="center",
                transform=ax_v.transAxes,
            )

    def generate_standard(self):
        print("🚀 Generando Dashboards Estándar (Pagina 1 y 2)...")
        for p_idx, page_rows in enumerate(PAGES):
            fig = plt.figure(figsize=(8.5, 11))
            gs = gridspec.GridSpec(
                len(page_rows),
                2,
                width_ratios=[1, 1.2],
                hspace=0.55,
                wspace=0.25,
                left=0.08,
                right=0.96,
                top=0.93,
                bottom=0.05,
            )
            fig.suptitle(
                f"IEEE Benchmark Scenarios — Physics Layer (Page {p_idx+1})",
                fontsize=12,
                fontweight="bold",
                y=0.98,
            )

            for i, row in enumerate(page_rows):
                self._render_row(
                    fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), row
                )

            base = os.path.join(OUTPUT_DIR, f"dashboard_physics_page_{p_idx+1}")
            fig.savefig(f"{base}.pdf")
            fig.savefig(f"{base}.png", dpi=600)
            plt.close(fig)

    def generate_compact(self):
        print("🚀 Generando Dashboard Compacto (Todo en una página)...")
        # Aplanar todas las filas en una sola lista
        all_rows = [row for page in PAGES for row in page]
        n_rows = len(all_rows)

        # Figura extendida verticalmente o muy apretada en Letter.
        # Para 13 filas en carta (11in), cada fila tiene <0.8in. Es muy apretado pero posible.
        fig = plt.figure(figsize=(8.5, 11))
        # hspace reducido para que quepa
        gs = gridspec.GridSpec(
            n_rows,
            2,
            width_ratios=[1, 1.2],
            hspace=0.6,
            wspace=0.25,
            left=0.06,
            right=0.98,
            top=0.96,
            bottom=0.03,
        )
        fig.suptitle(
            "IEEE Benchmark Scenarios — Physics Layer (Compact View)",
            fontsize=10,
            fontweight="bold",
            y=0.99,
        )

        for i, row in enumerate(all_rows):
            self._render_row(fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), row)

        base = os.path.join(OUTPUT_DIR, "dashboard_physics_compact")
        print(f"   💾 Guardando {base}...")
        fig.savefig(f"{base}.pdf")
        fig.savefig(f"{base}.png", dpi=600)
        fig.savefig(f"{base}.svg")
        plt.close(fig)

    def run(self):
        self.generate_standard()
        self.generate_compact()
        print("✅ Generación Completa.")


if __name__ == "__main__":
    DashboardGenerator().run()
