from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .export import save_fig


def _heatmap(Z, xvals, yvals, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[min(xvals), max(xvals), min(yvals), max(yvals)],
    )
    plt.colorbar(label="Score")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_path)


def plot_pll_landscape(out_dir: str, scenario: str, kp_vals, ki_vals, rmse_grid):
    _heatmap(
        Z=rmse_grid,
        xvals=kp_vals,
        yvals=ki_vals,
        xlabel="Kp",
        ylabel="Ki",
        title=f"{scenario} — PLL RMSE Landscape",
        out_path=f"{out_dir}/landscapes/{scenario}__pll_landscape",
    )


def plot_kf_landscape(out_dir: str, scenario: str, q_vals, r_vals, rmse_grid, label="KF"):
    _heatmap(
        Z=rmse_grid,
        xvals=q_vals,
        yvals=r_vals,
        xlabel="Q",
        ylabel="R",
        title=f"{scenario} — {label} RMSE Landscape",
        out_path=f"{out_dir}/landscapes/{scenario}__{label.lower()}_landscape",
    )


def plot_rls_landscape(out_dir: str, scenario: str, lam_vals, win_vals, rmse_grid):
    _heatmap(
        Z=rmse_grid,
        xvals=lam_vals,
        yvals=win_vals,
        xlabel="lambda",
        ylabel="win_smooth",
        title=f"{scenario} — RLS RMSE Landscape",
        out_path=f"{out_dir}/landscapes/{scenario}__rls_landscape",
    )
