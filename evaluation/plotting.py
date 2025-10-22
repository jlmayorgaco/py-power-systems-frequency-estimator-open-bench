from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes
from matplotlib.figure import Figure


def plot_signal_and_estimators(
    signal: np.ndarray,
    f_true: np.ndarray,
    estimates: Mapping[str, Sequence[float]],
    fs: float,
    title: str = "Scenario",
    zoom_windows_top: Iterable[tuple[float, float]] | None = None,
    zoom_window_bottom: tuple[float, float] | None = None,
) -> Figure:
    signal = np.asarray(signal, dtype=float).ravel()
    f_true = np.asarray(f_true, dtype=float).ravel()
    n = signal.shape[0]
    t = np.arange(n, dtype=float) / float(fs)

    fig = plt.figure(figsize=(6.0, 3.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

    if zoom_windows_top:
        bax = brokenaxes(xlims=list(zoom_windows_top), hspace=0.05, fig=fig, subplot_spec=gs[0])
        for t0, t1 in zoom_windows_top:
            mask = (t >= t0) & (t <= t1)
            bax.plot(t[mask], signal[mask], linewidth=1.0, label=f"{t0}-{t1}s")
        bax.set_ylabel("Amplitude", fontsize=9)
        bax.legend(fontsize=7, loc="best", framealpha=0.9)
        bax.set_title(f"{title} — AC Signal (zoomed ranges)", fontsize=9)
    else:
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(t, signal, linewidth=1.0)
        ax0.set_ylabel("Amplitude", fontsize=9)
        ax0.set_title(f"{title} — AC Signal (full)", fontsize=9)
        ax0.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax1 = fig.add_subplot(gs[1])

    t_end = float(t[-1]) if t.size else 0.0
    for name, f_hat in estimates.items():
        f_hat_arr = np.asarray(f_hat, dtype=float).ravel()
        t_est = (
            np.linspace(0.0, t_end, num=f_hat_arr.size)
            if f_hat_arr.size
            else np.array([], dtype=float)
        )
        ax1.plot(t_est, f_hat_arr, linewidth=1.0, label=f"{name} estimate")

    ax1.plot(t[: f_true.size], f_true, linestyle="--", linewidth=1.2, label="True Frequency")
    ax1.set_xlabel("Time [s]", fontsize=9)
    ax1.set_ylabel("Frequency [Hz]", fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9, loc="best")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.set_title("Actual vs Estimated Frequency", fontsize=9)

    if zoom_window_bottom is not None:
        t0, t1 = zoom_window_bottom
        ax1.set_xlim(t0, t1)
        mask = (t >= t0) & (t <= t1)
        if np.any(mask):
            f_ref = float(np.mean(f_true[: t.size][mask]))
            margin = 0.15 * f_ref
            ax1.set_ylim(f_ref - margin, f_ref + margin)

    plt.tight_layout()
    return fig
