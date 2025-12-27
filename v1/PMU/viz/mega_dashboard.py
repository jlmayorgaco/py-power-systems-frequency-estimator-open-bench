# dashboards/physics_dashboard.py
# Refinements: better zoom + delta braces/arrows for steps (E4/E5, E6/E7)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ============================================================
# IEEE-ish style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,              # antes 8
    "axes.labelsize": 9,         # antes 7
    "axes.titlesize": 9,         # antes 7
    "xtick.labelsize": 8,        # antes 7
    "ytick.labelsize": 8,        # antes 7
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "savefig.dpi": 600,
})

VOLT_COLOR = "#2E4053"
FREQ_COLOR = "#B03A2E"


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class DashboardStyle:
    volt_color: str = VOLT_COLOR
    freq_color: str = FREQ_COLOR
    lw_v: float = 0.7
    lw_f: float = 1.2
    tick_labelsize: int = 6
    nx_v: int = 4
    ny_v: int = 4
    nx_f: int = 5
    ny_f: int = 4
    title_fontsize: int = 7
    label_fontsize: int = 7
    title_pad: int = 2

    # reference waveform overlay
    ref60_color: str = "k"
    ref60_lw: float = 0.8
    ref60_ls: str = "--"
    ref60_alpha: float = 0.55

    # legends
    legend_fontsize: int = 6
    legend_frameon: bool = False

    # annotation styling
    anno_color: str = "k"
    anno_lw: float = 0.8
    anno_alpha: float = 0.85
    anno_fontsize: int = 7


@dataclass(frozen=True)
class LayoutConfig:
    # default dashboard (2 columns)
    scenarios_per_page: int = 8
    figsize: Tuple[float, float] = (10, 14)
    width_ratios: Tuple[float, float] = (1.0, 1.2)
    hspace: float = 0.45
    wspace: float = 0.18
    left: float = 0.07
    right: float = 0.99
    bottom: float = 0.045
    top: float = 0.95
    suptitle_y: float = 0.985

    # comparative dashboard (2 columns, grouped)
    comp_groups_per_page: int = 4
    comp_figsize: Tuple[float, float] = (10, 14)
    comp_width_ratios: Tuple[float, float] = (1.0, 1.2)
    comp_hspace: float = 0.50
    comp_wspace: float = 0.18
    comp_left: float = 0.07
    comp_right: float = 0.99
    comp_bottom: float = 0.045
    comp_top: float = 0.93
    comp_suptitle_y: float = 0.975


# ============================================================
# Utilities
# ============================================================
def robust_ylim(y: np.ndarray, pad_frac: float = 0.10, q: Tuple[float, float] = (1, 99)) -> Optional[List[float]]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    lo, hi = np.percentile(y, q)
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6
    pad = (hi - lo) * float(pad_frac)
    return [float(lo - pad), float(hi + pad)]


def event_window(
    t: np.ndarray,
    x: np.ndarray,
    default_center: float = 2.0,
    default_width: float = 0.10,
    width: Tuple[float, float] = (0.06, 0.25),
    thr_sigma: float = 4.0,
) -> List[float]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    if t.size < 10:
        return [default_center - default_width / 2, default_center + default_width / 2]

    dt = np.diff(t)
    dx = np.diff(x)
    dt[dt == 0] = np.nan

    g = np.abs(dx / dt)
    g = g[np.isfinite(g)]
    if g.size < 5:
        return [default_center - default_width / 2, default_center + default_width / 2]

    med = np.median(g)
    mad = np.median(np.abs(g - med)) + 1e-12

    score = (np.abs(np.diff(x) / np.diff(t)) - med) / mad
    score = np.where(np.isfinite(score), score, 0.0)

    idx = int(np.argmax(score))
    if score[idx] < thr_sigma:
        return [default_center - default_width / 2, default_center + default_width / 2]

    t_event = float(t[idx])
    w = float(np.clip(default_width, width[0], width[1]))
    t0, t1 = t_event - w / 2, t_event + w / 2

    t0 = max(t0, float(t.min()))
    t1 = min(t1, float(t.max()))

    if (t1 - t0) < width[0]:
        mid = 0.5 * (t0 + t1)
        t0 = max(mid - width[0] / 2, float(t.min()))
        t1 = min(mid + width[0] / 2, float(t.max()))

    return [float(t0), float(t1)]


# ============================================================
# Scenario config (you can keep tweaking these)
# Key: comparative rows should use tight windows.
# ============================================================
def get_scenario_config() -> Dict[str, Dict[str, Any]]:
    def_v = [1.95, 2.05]
    def_f = [0.0, 5.0]
    return {
        "G1_E1_Pure_60Hz": {"t_v": def_v, "t_f": def_f, "f_lims": [59.9, 60.1]},
        "G1_E2_Gaussian_Noise_1pct": {"t_v": def_v, "t_f": def_f, "f_lims": [59.5, 60.5]},
        "G1_E3_Gaussian_Noise_5pct": {"t_v": [1.98, 2.02], "t_f": def_f, "f_lims": [58.5, 61.5]},

        # Voltage steps: show a couple of cycles around t=2 (tight)
        "G2_E4_Voltage_Mag_Step_1pct": {"t_v": [1.995, 2.020], "t_f": [1.98, 2.02], "f_lims": [59.9, 60.1]},
        "G2_E5_Voltage_Mag_Step_10pct": {"t_v": [1.995, 2.020], "t_f": [1.98, 2.02], "f_lims": [59.9, 60.1]},

        # Frequency steps: tight around t=2
        "G2_E6_Freq_Step_60_to_59p5": {"t_v": [1.98, 2.02], "t_f": [1.98, 2.02], "f_lims": [59.4, 60.1]},
        "G2_E7_Freq_Step_60_to_55": {"t_v": [1.98, 2.02], "t_f": [1.98, 2.02], "f_lims": [54.5, 60.5]},

        "G2_E8_Fast_Ramp_plus5Hzs": {"t_v": [1.5, 5.0], "t_f": [1.5, 5.0], "f_lims": [59.5, 80.5]},
        "G2_E9_Slow_Ramp_minus0p5Hzs": {"t_v": def_v, "t_f": [0.5, 5.0], "f_lims": [57.0, 60.5]},

        # -----------------------------
        # GROUP 3: ROBUSTNESS & MODULATIONS
        # -----------------------------
        "G3_E10_AM_Modulation": {
            "t_v": [0.0, 2.0],       # AM se aprecia bien en 1-2s
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],  # f ~ constante (solo detune microHz)
        },

        "G3_E11_FM_Modulation": {
            "t_v": [0.0, 1.2],       # v(t) cambia “fase” por FM
            "t_f": [0.0, 5.0],
            "f_lims": [58.5, 61.5],  # 60 ± 1 Hz
        },

        "G3_E12_Composite_Islanding": {
            # evento real: phase jump en t=1.0
            "t_v": [0.95, 1.05],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },

        "G3_E13_Impulsive_Outliers": {
            # impulsos aleatorios → mejor vista global (y robust_ylim hace su magia)
            "t_v": [0.0, 5.0],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },

        "G3_E14_Noise_Harmonics": {
            "t_v": [0.0, 0.25],      # zoom corto para ver distorsión armónica
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },

        "G3_E15_Multi_Event_Profile": {
            # ramp de t=1 a t=3, luego hold
            "t_v": [0.9, 3.4],
            "t_f": [0.9, 3.4],
            "f_lims": [59.5, 64.8],  # llega a +4 Hz (≈64)
        },

        "G4_E16_Chamorro_Event": {"t_v": [2.40, 2.60], "t_f": def_f, "f_lims": [59.5, 60.5]},
    }


def parse_label(sc_id: str) -> Tuple[str, str]:
    parts = sc_id.split("_")
    code = f"{parts[0]}-{parts[1]}"
    desc = " ".join(parts[2:]).replace("plus", "+").replace("p", ".").replace("pct", "%")
    return code, desc


# ============================================================
# Plotter (base)
# ============================================================
class PhysicsDashboardPlotter:
    def __init__(
        self,
        base_path: str = "artifacts/results_mc/waveforms",
        out_dir: str = "artifacts/dashboards",
        style: DashboardStyle = DashboardStyle(),
        layout: LayoutConfig = LayoutConfig(),
        conf_master: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.base_path = base_path
        self.out_dir = out_dir
        self.style = style
        self.layout = layout
        self.conf_master = conf_master if conf_master is not None else get_scenario_config()
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------- IO ----------
    def _csv_path(self, sc_id: str) -> str:
        return os.path.join(self.base_path, sc_id, "ground_truth.csv")

    def _load_waveform(self, sc_id: str) -> Optional[pd.DataFrame]:
        path = self._csv_path(sc_id)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        for c in ("t", "real_v", "real_f"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["t", "real_v", "real_f"])
        if df.empty:
            return None
        return df

    # ---------- ticks / labels ----------
    def _force_ticks(self, ax, nx: int, ny: int) -> None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=self.style.tick_labelsize,
            direction="out",
            length=3.0,
            width=0.6,
            pad=2.0,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labelleft=True,
        )

    def _maybe_hide_xlabel(self, ax, is_last_row: bool) -> None:
        if is_last_row:
            ax.set_xlabel("Time [s]", fontsize=self.style.label_fontsize)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    # ---------- config logic ----------
    def _auto_conf(self, df: pd.DataFrame, sc_id: str) -> Dict[str, Any]:
        manual = dict(self.conf_master.get(sc_id, {}))
        t = df["t"].to_numpy()
        v = df["real_v"].to_numpy()
        f = df["real_f"].to_numpy()

        if "t_v" not in manual:
            manual["t_v"] = event_window(t, v, default_center=2.0, default_width=0.10, width=(0.06, 0.20))
        if "t_f" not in manual:
            manual["t_f"] = event_window(t, f, default_center=2.0, default_width=3.0, width=(2.0, 5.0))

        manual["v_lims"] = robust_ylim(v, pad_frac=0.10) or [-1.2, 1.2]
        if "f_lims" not in manual:
            manual["f_lims"] = robust_ylim(f, pad_frac=0.10) or [59.5, 60.5]

        return manual

    # ---------- reference waveform ----------
    @staticmethod
    def _ideal_60hz(t: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * 60.0 * t)

    def _overlay_60hz(self, ax, t: np.ndarray, label: Optional[str] = None) -> None:
        ax.plot(
            t,
            self._ideal_60hz(t),
            color=self.style.ref60_color,
            lw=self.style.ref60_lw,
            linestyle=self.style.ref60_ls,
            alpha=self.style.ref60_alpha,
            label=label,
            zorder=2,
        )

    # ---------- annotations helpers ----------
    def _bracket_delta(
        self,
        ax,
        x: float,
        y0: float,
        y1: float,
        text: str,
        x_offset_frac: float = 0.015,
    ) -> None:
        """
        Draw a vertical bracket (|-|) with text next to it.
        Uses axes-relative x offset so it stays visible when zoom changes.
        """
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        dx = (xmax - xmin) * x_offset_frac
        x_br = x + dx

        ax.annotate(
            "",
            xy=(x_br, y1),
            xytext=(x_br, y0),
            arrowprops=dict(
                arrowstyle="|-|",
                color=self.style.anno_color,
                lw=self.style.anno_lw,
                alpha=self.style.anno_alpha,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )

        y_mid = 0.5 * (y0 + y1)
        ax.text(
            x_br + dx * 0.6,
            y_mid,
            text,
            fontsize=self.style.anno_fontsize,
            color=self.style.anno_color,
            alpha=self.style.anno_alpha,
            va="center",
            ha="left",
            zorder=5,
        )

    # ---------- scenario-specific annotations ----------
    def _annotate_voltage_step_delta(self, ax_v, sc_id: str) -> None:
        """
        For E4/E5: show ΔA in % (amplitude step) near the top of voltage axis.
        (Your generator: a=1 -> 0.99 or 0.90 at t=2.0)
        """
        if sc_id not in ("G2_E4_Voltage_Mag_Step_1pct", "G2_E5_Voltage_Mag_Step_10pct"):
            return

        t_step = 2.0
        a_pre = 1.0
        a_post = 0.99 if "1pct" in sc_id else 0.90
        delta_pct = 100.0 * (a_post - a_pre)  # negative

        # place bracket in voltage axis coordinates (near upper portion)
        ymin, ymax = ax_v.get_ylim()
        y0 = ymin + 0.78 * (ymax - ymin)
        y1 = ymin + 0.93 * (ymax - ymin)

        ax_v.axvline(t_step, color="k", linestyle=":", linewidth=0.8, alpha=0.8, zorder=3)

        txt = rf"$\Delta A = {delta_pct:+.0f}\%$"
        self._bracket_delta(ax_v, x=t_step, y0=y0, y1=y1, text=txt)

    def _annotate_frequency_step_delta(self, ax_f, sc_id: str) -> None:
        """
        For E6/E7: show Δf in Hz and % at the frequency step in f(t) subplot.
        (Your generator: f = f0 before 2.0, f0-0.5 or f0-5.0 after)
        """
        if sc_id not in ("G2_E6_Freq_Step_60_to_59p5", "G2_E7_Freq_Step_60_to_55"):
            return

        t_step = 2.0
        f_pre = 60.0
        f_post = 59.5 if "59p5" in sc_id else 55.0
        df = f_post - f_pre
        df_pct = 100.0 * (df / f_pre)

        ax_f.axvline(t_step, color="k", linestyle=":", linewidth=0.8, alpha=0.8, zorder=3)

        # bracket between the two levels (place it near the right side of t_step)
        y0, y1 = (f_post, f_pre) if f_post < f_pre else (f_pre, f_post)

        txt = rf"$\Delta f = {df:+.1f}\,\mathrm{{Hz}}$  ({df_pct:+.2f}\%)"
        self._bracket_delta(ax_f, x=t_step, y0=y0, y1=y1, text=txt)

    # ---------- default dashboard plots ----------
    def _plot_voltage(self, fig, gs, row_idx: int, df: pd.DataFrame, sc_id: str, conf: Dict[str, Any], is_last_row: bool) -> None:
        code, _desc = parse_label(sc_id)
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.plot(df["t"], df["real_v"], color=self.style.volt_color, lw=self.style.lw_v, zorder=3)

        ax.set_xlim(conf["t_v"])
        ax.set_ylim(conf["v_lims"])
        ax.set_ylabel("v(t) [pu]", fontsize=self.style.label_fontsize)
        self._maybe_hide_xlabel(ax, is_last_row)

        ax.set_title(f"{code}: Voltage", pad=self.style.title_pad, fontsize=self.style.title_fontsize)

        # annotate voltage steps with delta
        self._annotate_voltage_step_delta(ax, sc_id)

        self._force_ticks(ax, nx=self.style.nx_v, ny=self.style.ny_v)

    def _plot_frequency(self, fig, gs, row_idx: int, df: pd.DataFrame, sc_id: str, conf: Dict[str, Any], is_last_row: bool) -> None:
        _code, desc = parse_label(sc_id)
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.plot(df["t"], df["real_f"], color=self.style.freq_color, lw=self.style.lw_f)

        ax.set_xlim(conf["t_f"])
        ax.set_ylim(conf["f_lims"])
        ax.set_ylabel("f [Hz]", fontsize=self.style.label_fontsize)
        self._maybe_hide_xlabel(ax, is_last_row)

        ax.set_title(f"Frequency: {desc}", pad=self.style.title_pad, fontsize=self.style.title_fontsize)

        # annotate frequency steps with delta
        self._annotate_frequency_step_delta(ax, sc_id)

        self._force_ticks(ax, nx=self.style.nx_f, ny=self.style.ny_f)

    @staticmethod
    def _batches(items: List[str], batch_size: int) -> List[List[str]]:
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _make_figure(self, nrows: int) -> Tuple[plt.Figure, gridspec.GridSpec]:
        fig = plt.figure(figsize=self.layout.figsize)
        gs = gridspec.GridSpec(
            nrows,
            2,
            figure=fig,
            width_ratios=list(self.layout.width_ratios),
            hspace=self.layout.hspace,
            wspace=self.layout.wspace,
        )
        return fig, gs

    def _finalize_and_save(self, fig: plt.Figure, page_idx: int) -> None:
        fig.suptitle(
            f"Power Systems Ground Truth - Physics Layer (Page {page_idx})",
            y=self.layout.suptitle_y,
            fontsize=12,
            fontweight="bold",
        )
        fig.subplots_adjust(left=self.layout.left, right=self.layout.right, bottom=self.layout.bottom, top=self.layout.top)

        base = os.path.join(self.out_dir, f"Dashboard_Physics_P{page_idx}")
        fig.savefig(f"{base}.pdf", format="pdf")
        fig.savefig(f"{base}.png", format="png", dpi=300)
        plt.close(fig)

    def run(self) -> None:
        sc_list = list(self.conf_master.keys())
        batches = self._batches(sc_list, self.layout.scenarios_per_page)

        for p_idx, batch in enumerate(batches, start=1):
            fig, gs = self._make_figure(nrows=len(batch))

            for row_idx, sc_id in enumerate(batch):
                df = self._load_waveform(sc_id)
                if df is None:
                    continue
                conf = self._auto_conf(df, sc_id)
                is_last_row = (row_idx == len(batch) - 1)

                self._plot_voltage(fig, gs, row_idx, df, sc_id, conf, is_last_row)
                self._plot_frequency(fig, gs, row_idx, df, sc_id, conf, is_last_row)

            self._finalize_and_save(fig, page_idx=p_idx)

        print(f"✅ Physics dashboard (per-scenario) generado en {self.out_dir}")

class ComparativePhysicsDashboardPlotter(PhysicsDashboardPlotter):
    """
    Same aesthetics, grouped rows, plus Δ annotations:
      - Row1: G1 (ordered) + custom legend with Noise %
      - Row2: E4/E5 (+ ref60 on voltage + legend ΔV% + arrow to each peak ON VOLTAGE CURVE)
      - Row3: E6/E7 (+ ref60 on voltage + legend Δf + arrow to post-step plateau ON VOLTAGE (phase accel))
      - Row4: single paginated
    """

    ROW_SPECS_FIXED: List[Dict[str, Any]] = [
        {
            "scenario_ids": ["G1_E3_Gaussian_Noise_5pct", "G1_E2_Gaussian_Noise_1pct", "G1_E1_Pure_60Hz"],
            "title": "Group G1 (E3 → E2 → E1)",
            "show_ref60": False,
            "x": {"t_v": [1.98, 2.02], "t_f": [1.95, 2.05]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": ["G2_E5_Voltage_Mag_Step_10pct", "G2_E4_Voltage_Mag_Step_1pct"],
            "title": "Group G2 (E5 → E4) Voltage Steps",
            "show_ref60": True,
            "x": {"t_v": [1.998, 2.016], "t_f": [1.98, 2.02]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": ["G2_E7_Freq_Step_60_to_55", "G2_E6_Freq_Step_60_to_59p5"],
            "title": "Group G2 (E7 → E6) Frequency Steps",
            "show_ref60": True,
            "x": {"t_v": [1.998, 2.016], "t_f": [1.98, 2.02]},
            "y": {"f_lims": [54.5, 60.5]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": ["G2_E8_Fast_Ramp_plus5Hzs", "G2_E9_Slow_Ramp_minus0p5Hzs"],
            "title": "Group G2 (E8 → E9) Frequency Ramps",
            "show_ref60": True,
            "x": {"t_v": [1.998, 2.016], "t_f": [0.5, 5.0]},
            "y": {"f_lims": [57.0, 80.5]},
            "legend": {"loc": "upper right"},
        },
    ]

    # Add here any other singles you want paginated at the bottom row
    ROW_SPECS_SINGLES: List[Dict[str, Any]] = [
        {"scenario_ids": ["G3_E10_AM_Modulation"], "title": "Single: G3-E10 AM", "show_ref60": True,
        "x": {"t_v": [0.0, 2.0], "t_f": [0.0, 5.0]}, "y": {"f_lims": [59.9, 60.1]}},

        {"scenario_ids": ["G3_E11_FM_Modulation"], "title": "Single: G3-E11 FM", "show_ref60": True,
        "x": {"t_v": [0.0, 1.2], "t_f": [0.0, 5.0]}, "y": {"f_lims": [58.5, 61.5]}},

        {"scenario_ids": ["G3_E12_Composite_Islanding"], "title": "Single: G3-E12 Composite Islanding", "show_ref60": True,
        "x": {"t_v": [0.95, 1.05], "t_f": [0.0, 5.0]}, "y": {"f_lims": [59.9, 60.1]}},

        {"scenario_ids": ["G3_E13_Impulsive_Outliers"], "title": "Single: G3-E13 Impulsive Outliers", "show_ref60": True,
        "x": {"t_v": [1.85, 2.25], "t_f": [1.85, 2.25]}, "y": {"f_lims": [59.9, 60.1]}},

        {"scenario_ids": ["G3_E14_Noise_Harmonics"], "title": "Single: G3-E14 Harmonics + Noise", "show_ref60": True,
        "x": {"t_v": [0.0, 0.25], "t_f": [0.0, 5.0]}, "y": {"f_lims": [59.9, 60.1]}},

        {"scenario_ids": ["G3_E15_Multi_Event_Profile"], "title": "Single: G3-E15 Multi-Event", "show_ref60": True,
        "x": {"t_v": [0.0, 5.0], "t_f": [0.0, 5.0]}, "y": {"f_lims": [59.5, 64.8]}},

        {"scenario_ids": ["G4_E16_Chamorro_Event"], "title": "Single: G4-E16 Chamorro", "show_ref60": False,
        "x": {"t_v": [2.40, 2.60], "t_f": [0.0, 5.0]}, "y": {"f_lims": [59.5, 60.5]}},
    ]

    # -----------------------------
    # limits helpers
    # -----------------------------
    def _apply_manual_limits(self, ax_v, ax_f, row_spec: Dict[str, Any], confs: List[Dict[str, Any]]) -> None:
        x = row_spec.get("x", {}) or {}
        y = row_spec.get("y", {}) or {}

        if "t_v" in x:
            ax_v.set_xlim(list(x["t_v"]))
        elif confs and "t_v" in confs[0]:
            ax_v.set_xlim(list(confs[0]["t_v"]))

        if "t_f" in x:
            ax_f.set_xlim(list(x["t_f"]))
        elif confs and "t_f" in confs[0]:
            ax_f.set_xlim(list(confs[0]["t_f"]))

        if "v_lims" in y:
            ax_v.set_ylim(list(y["v_lims"]))
        else:
            ax_v.set_ylim(self._merge_ylims(confs, "v_lims", fallback=[-1.2, 1.2]))

        if "f_lims" in y:
            ax_f.set_ylim(list(y["f_lims"]))
        else:
            ax_f.set_ylim(self._merge_ylims(confs, "f_lims", fallback=[59.5, 60.5]))

    @staticmethod
    def _merge_ylims(confs: List[Dict[str, Any]], key: str, fallback: List[float]) -> List[float]:
        lows, highs = [], []
        for c in confs:
            if key in c and c[key] is not None:
                lows.append(float(c[key][0]))
                highs.append(float(c[key][1]))
        if not lows or not highs:
            return fallback
        return [min(lows), max(highs)]

    # -----------------------------
    # group classifiers (set-based, order-independent)
    # IMPORTANT: call these with sc_ok (actually plotted scenarios).
    # -----------------------------
    @staticmethod
    def _is_g1_group_exact(group: List[str]) -> bool:
        return set(group) == {
            "G1_E1_Pure_60Hz",
            "G1_E2_Gaussian_Noise_1pct",
            "G1_E3_Gaussian_Noise_5pct",
        }

    @staticmethod
    def _is_e45_group_exact(group: List[str]) -> bool:
        return set(group) == {
            "G2_E4_Voltage_Mag_Step_1pct",
            "G2_E5_Voltage_Mag_Step_10pct",
        }

    @staticmethod
    def _is_e67_group_exact(group: List[str]) -> bool:
        return set(group) == {
            "G2_E6_Freq_Step_60_to_59p5",
            "G2_E7_Freq_Step_60_to_55",
        }

    @staticmethod
    def _is_e89_group_exact(group: List[str]) -> bool:
        return set(group) == {
            "G2_E8_Fast_Ramp_plus5Hzs",
            "G2_E9_Slow_Ramp_minus0p5Hzs",
        }

    # -----------------------------
    # legend label builders (KEEP ORDER = plotting order)
    # -----------------------------
    def _make_g1_legend_labels(self, group: List[str]) -> List[str]:
        noise_map = {
            "G1_E1_Pure_60Hz": "Noise = 0%",
            "G1_E2_Gaussian_Noise_1pct": "Noise = 1%",
            "G1_E3_Gaussian_Noise_5pct": "Noise = 5%",
        }
        return [noise_map.get(sc_id, parse_label(sc_id)[0]) for sc_id in group]

    def _make_e45_legend_labels(self, group: List[str]) -> List[str]:
        dv_map = {
            "G2_E4_Voltage_Mag_Step_1pct": -1.0,
            "G2_E5_Voltage_Mag_Step_10pct": -10.0,
        }
        out: List[str] = []
        for sc_id in group:
            dv = dv_map.get(sc_id, None)
            out.append(parse_label(sc_id)[0] if dv is None else f"ΔV = {dv:.0f}%")
        return out

    def _make_e67_legend_labels(self, group: List[str]) -> List[str]:
        df_map = {
            "G2_E6_Freq_Step_60_to_59p5": -0.5,
            "G2_E7_Freq_Step_60_to_55": -5.0,
        }
        out: List[str] = []
        for sc_id in group:
            df = df_map.get(sc_id, None)
            out.append(parse_label(sc_id)[0] if df is None else f"Δf = {df:+.1f} Hz")
        return out

    def _make_e89_legend_labels(self, group: List[str]) -> List[str]:
        rocof_map = {
            "G2_E8_Fast_Ramp_plus5Hzs": "+5.0 Hz/s",
            "G2_E9_Slow_Ramp_minus0p5Hzs": "-0.5 Hz/s",
        }
        out: List[str] = []
        for sc_id in group:
            r = rocof_map.get(sc_id, None)
            out.append(parse_label(sc_id)[0] if r is None else f"RoCoF = {r}")
        return out

    # -----------------------------
    # E4/E5: arrow anchored ON VOLTAGE CURVE (post-step peak)
    # -----------------------------
    def _annotate_voltage_curve_delta(self, ax, t: np.ndarray, v: np.ndarray, sc_id: str) -> None:
        if sc_id not in ("G2_E4_Voltage_Mag_Step_1pct", "G2_E5_Voltage_Mag_Step_10pct"):
            return

        dv_map = {
            "G2_E4_Voltage_Mag_Step_1pct": -1.0,
            "G2_E5_Voltage_Mag_Step_10pct": -10.0,
        }
        dv = dv_map.get(sc_id, None)
        if dv is None:
            return
        text = f"ΔV = {dv:.0f}%"

        x0, x1 = ax.get_xlim()
        t_step = 2.0

        m = (t >= max(t_step, x0)) & (t <= x1) & np.isfinite(t) & np.isfinite(v)
        if not np.any(m):
            return
        tt, vv = t[m], v[m]
        k = int(np.argmax(vv))
        t_anchor = float(tt[k])
        v_anchor = float(vv[k])

        y0, y1 = ax.get_ylim()
        dx = 0.10 * (x1 - x0)
        dy = -0.10 * (y1 - y0)

        ax.annotate(
            text,
            xy=(t_anchor, v_anchor),
            xytext=(t_anchor + dx, v_anchor + dy),
            textcoords="data",
            fontsize=self.style.anno_fontsize,
            ha="left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                lw=self.style.anno_lw,
                color=self.style.anno_color,
                shrinkA=0,
                shrinkB=0,
                alpha=self.style.anno_alpha,
            ),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            zorder=7,
        )

    # -----------------------------
    # E6/E7: annotate Δf BUT draw arrow on VOLTAGE subplot (phase accel)
    # -----------------------------
    def _annotate_voltage_phase_delta_from_df(
        self,
        ax,
        t: np.ndarray,
        v: np.ndarray,
        sc_id: str,
        t_step: float = 2.01,
    ) -> None:
        if sc_id not in ("G2_E6_Freq_Step_60_to_59p5", "G2_E7_Freq_Step_60_to_55"):
            return

        f_pre = 60.0
        f_post = 59.5 if "59p5" in sc_id else 55.0
        df_hz = f_post - f_pre
        df_pct = 100.0 * (df_hz / f_pre)
        text = rf"$\Delta f = {df_hz:+.1f}\,\mathrm{{Hz}}\;({df_pct:+.2f}\%)$"

        x0, x1 = ax.get_xlim()
        t_target = max(t_step + 0.15 * (x1 - x0), t_step + 1e-6)
        t_target = min(t_target, x1 - 1e-6)

        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        m = np.isfinite(t) & np.isfinite(v) & (t >= x0) & (t <= x1)
        if not np.any(m):
            return
        tt = t[m]
        vv = v[m]
        k = int(np.argmin(np.abs(tt - t_target)))
        t_anchor = float(tt[k])
        v_anchor = float(vv[k])

        y0, y1 = ax.get_ylim()
        dx = 0.10 * (x1 - x0)
        dy = 0.10 * (y1 - y0)

        ax.annotate(
            text,
            xy=(t_anchor, v_anchor),
            xytext=(t_anchor + dx, v_anchor + dy),
            textcoords="data",
            fontsize=self.style.anno_fontsize,
            ha="left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                lw=self.style.anno_lw,
                color=self.style.anno_color,
                shrinkA=0,
                shrinkB=0,
                alpha=self.style.anno_alpha,
            ),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            zorder=7,
        )

    # -----------------------------
    # main row plot
    # -----------------------------
    def _plot_row(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        row_spec: Dict[str, Any],
        is_last_row: bool,
    ) -> None:
        group = list(row_spec["scenario_ids"])
        title = str(row_spec.get("title", "Grouped"))
        show_ref60 = bool(row_spec.get("show_ref60", False))

        dfs: List[pd.DataFrame] = []
        confs: List[Dict[str, Any]] = []
        codes: List[str] = []
        sc_ok: List[str] = []

        # --- load (keep alignment between sc_id and df) ---
        for sc_id in group:
            df = self._load_waveform(sc_id)
            if df is None:
                continue
            sc_ok.append(sc_id)
            dfs.append(df)
            confs.append(self._auto_conf(df, sc_id))
            code, _ = parse_label(sc_id)
            codes.append(code)

        ax_v = fig.add_subplot(gs[row_idx, 0])
        ax_f = fig.add_subplot(gs[row_idx, 1])

        ref_label_used = False

        # --- plot overlays ---
        for sc_id, df in zip(sc_ok, dfs):
            code, _desc = parse_label(sc_id)

            ax_v.plot(df["t"], df["real_v"], color=self.style.volt_color, lw=0.9, label=code, zorder=3)
            ax_f.plot(df["t"], df["real_f"], color=self.style.freq_color, lw=1.1, label=code, zorder=3)

            if show_ref60:
                lbl = None if ref_label_used else "Ideal 60 Hz"
                self._overlay_60hz(ax_v, df["t"].to_numpy(dtype=float), label=lbl)
                if lbl is not None:
                    ref_label_used = True

            # base annotations (safe)
            if hasattr(self, "_annotate_voltage_step_delta"):
                try:
                    self._annotate_voltage_step_delta(ax_v, sc_id)
                except Exception:
                    pass
            if hasattr(self, "_annotate_frequency_step_delta"):
                try:
                    self._annotate_frequency_step_delta(ax_f, sc_id)
                except Exception:
                    pass

        # --- labels / titles ---
        ax_v.set_ylabel("v(t) [pu]", fontsize=self.style.label_fontsize)
        ax_f.set_ylabel("f [Hz]", fontsize=self.style.label_fontsize)
        self._maybe_hide_xlabel(ax_v, is_last_row)
        self._maybe_hide_xlabel(ax_f, is_last_row)

        codes_str = " / ".join(codes) if codes else "N/A"
        ax_v.set_title(f"{title} — Voltage ({codes_str})", fontsize=self.style.title_fontsize, pad=self.style.title_pad)
        ax_f.set_title(f"{title} — Frequency ({codes_str})", fontsize=self.style.title_fontsize, pad=self.style.title_pad)

        # --- apply zoom BEFORE arrow annotations ---
        self._apply_manual_limits(ax_v, ax_f, row_spec=row_spec, confs=confs)

        # --- group-specific arrow annotations ---
        # IMPORTANT: check membership using sc_ok (actually plotted), not raw group.
        if self._is_e45_group_exact(sc_ok):
            for sc_id, df in zip(sc_ok, dfs):
                self._annotate_voltage_curve_delta(
                    ax_v,
                    t=df["t"].to_numpy(dtype=float),
                    v=df["real_v"].to_numpy(dtype=float),
                    sc_id=sc_id,
                )

        if self._is_e67_group_exact(sc_ok):
            for sc_id, df in zip(sc_ok, dfs):
                self._annotate_voltage_phase_delta_from_df(
                    ax_v,
                    t=df["t"].to_numpy(dtype=float),
                    v=df["real_v"].to_numpy(dtype=float),
                    sc_id=sc_id,
                    t_step=2.005,
                )

        # NOTE: do NOT add extra ref60 overlay here.
        # ref60 is already handled by show_ref60 inside the plot loop.

        # --- ticks ---
        self._force_ticks(ax_v, nx=self.style.nx_v, ny=self.style.ny_v)
        self._force_ticks(ax_f, nx=self.style.nx_f, ny=self.style.ny_f)

        # --- legends ---
        if len(codes) > 1:
            handles, _labels = ax_v.get_legend_handles_labels()
            leg_cfg = row_spec.get("legend", {}) or {}
            loc = leg_cfg.get("loc", "upper right")
            bbox = leg_cfg.get("bbox_to_anchor", None)

            if self._is_g1_group_exact(sc_ok):
                labels = self._make_g1_legend_labels(sc_ok)
                ax_v.legend(
                    handles,
                    labels,
                    fontsize=self.style.legend_fontsize,
                    frameon=self.style.legend_frameon,
                    ncol=1,
                    loc=loc,
                    bbox_to_anchor=bbox,
                    borderaxespad=0.0 if bbox else 0.5,
                    handlelength=2.0,
                    labelspacing=0.3,
                )
            elif self._is_e45_group_exact(sc_ok):
                labels = self._make_e45_legend_labels(sc_ok)
                ax_v.legend(
                    handles,
                    labels,
                    fontsize=self.style.legend_fontsize,
                    frameon=self.style.legend_frameon,
                    ncol=1,
                    loc=loc,
                    bbox_to_anchor=bbox,
                    borderaxespad=0.0 if bbox else 0.5,
                    handlelength=2.0,
                    labelspacing=0.3,
                )
            elif self._is_e67_group_exact(sc_ok):
                labels = self._make_e67_legend_labels(sc_ok)
                ax_v.legend(
                    handles,
                    labels,
                    fontsize=self.style.legend_fontsize,
                    frameon=self.style.legend_frameon,
                    ncol=1,
                    loc=loc,
                    bbox_to_anchor=bbox,
                    borderaxespad=0.0 if bbox else 0.5,
                    handlelength=2.0,
                    labelspacing=0.3,
                )
            elif self._is_e89_group_exact(sc_ok):
                labels = self._make_e89_legend_labels(sc_ok)
                ax_v.legend(
                    handles,
                    labels,
                    fontsize=self.style.legend_fontsize,
                    frameon=self.style.legend_frameon,
                    ncol=1,
                    loc=loc,
                    bbox_to_anchor=bbox,
                    borderaxespad=0.0 if bbox else 0.5,
                    handlelength=2.0,
                    labelspacing=0.3,
                )
            else:
                ax_v.legend(
                    fontsize=self.style.legend_fontsize,
                    frameon=self.style.legend_frameon,
                    loc="upper right",
                    ncol=2,
                )

    # -----------------------------
    # main (FIXED PAGINATION)
    # -----------------------------
    def run(self) -> None:
        # 1. Unified list
        all_specs = self.ROW_SPECS_FIXED + self.ROW_SPECS_SINGLES
        
        # 2. Linear pagination (no repetition)
        rows_per_page = 6 
        batches = [all_specs[i : i + rows_per_page] for i in range(0, len(all_specs), rows_per_page)]

        for p_idx, batch in enumerate(batches, start=1):
            nrows = len(batch)
            fig = plt.figure(figsize=self.layout.comp_figsize)
            gs = gridspec.GridSpec(
                nrows,
                2,
                figure=fig,
                width_ratios=list(self.layout.comp_width_ratios),
                hspace=self.layout.comp_hspace,
                wspace=self.layout.comp_wspace,
            )

            for row_idx, row_spec in enumerate(batch):
                is_last_row = (row_idx == nrows - 1)
                self._plot_row(fig, gs, row_idx, row_spec, is_last_row=is_last_row)

            fig.suptitle(
                f"Comparative Physics Dashboard (Grouped) — Page {p_idx}",
                y=self.layout.comp_suptitle_y,
                fontsize=12,
                fontweight="bold",
            )
            fig.subplots_adjust(
                left=self.layout.comp_left,
                right=self.layout.comp_right,
                bottom=self.layout.comp_bottom,
                top=self.layout.comp_top,
            )

            base = os.path.join(self.out_dir, f"Dashboard_Physics_Comparative_P{p_idx}")
            fig.savefig(f"{base}.pdf", format="pdf")
            fig.savefig(f"{base}.png", format="png", dpi=300)
            plt.close(fig)

        print(f"✅ Comparative dashboard generado en {self.out_dir} (Sin duplicados)")



# ============================================================
# Entry points
# ============================================================
def create_physics_dashboard() -> None:
    # If your manager calls this, you get both outputs.
    PhysicsDashboardPlotter().run()
    ComparativePhysicsDashboardPlotter().run()


def create_comparative_physics_dashboard() -> None:
    ComparativePhysicsDashboardPlotter().run()


if __name__ == "__main__":
    create_physics_dashboard()


def create_dynamic_dashboard():
    pass


def create_global_dashboard():
    pass
