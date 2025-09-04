# evaluation/plotting.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from brokenaxes import brokenaxes

def plot_signal_and_estimators(signal, f_true, estimates, fs,
                               title="Scenario",
                               zoom_windows_top=None,
                               zoom_window_bottom=None):
    """
    Plot with broken top axis (signal) and zoomed bottom axis (frequency).
    """
    t = np.arange(len(signal)) / fs

    # ---------------- Create figure and GridSpec ----------------
    fig = plt.figure(figsize=(6, 3.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

    # ---------------- Top subplot (signal) ----------------
    if zoom_windows_top:
        bax = brokenaxes(
            xlims=zoom_windows_top,
            hspace=0.05,
            fig=fig,
            subplot_spec=gs[0]
        )
        for (t0, t1) in zoom_windows_top:
            mask = (t >= t0) & (t <= t1)
            bax.plot(t[mask], signal[mask], lw=1, label=f"{t0}-{t1}s")
        bax.set_ylabel("Amplitude", fontsize=9)
        bax.legend(fontsize=7, loc="best", framealpha=0.9)
        bax.set_title("AC Signal (zoomed ranges)", fontsize=9)
    else:
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(t, signal, color="b", lw=1)
        ax0.set_ylabel("Amplitude", fontsize=9)
        ax0.set_title("AC Signal (full)", fontsize=9)
        ax0.grid(True, which="both", linestyle="--", linewidth=0.5)

    # ---------------- Bottom subplot (frequency) ----------------
    ax1 = fig.add_subplot(gs[1])
    for name, f_hat in estimates.items():
        t_est = np.linspace(0, t[-1], len(f_hat))
        ax1.plot(t_est, f_hat, lw=1, label=f"{name} estimate")

    ax1.plot(t[:len(f_true)], f_true, "k--", lw=1.2, label="True Frequency")
    ax1.set_xlabel("Time [s]", fontsize=9)
    ax1.set_ylabel("Frequency [Hz]", fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9, loc="best")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.set_title("Actual vs Estimated Frequency", fontsize=9)

    # ---------------- Apply zoom to bottom ----------------
    if zoom_window_bottom:
        t0, t1 = zoom_window_bottom
        ax1.set_xlim(t0, t1)
        mask = (t >= t0) & (t <= t1)
        if np.any(mask):
            f_ref = np.mean(f_true[mask])
            margin = 0.15 * f_ref
            ax1.set_ylim(f_ref - margin, f_ref + margin)

    plt.tight_layout()
    return fig
