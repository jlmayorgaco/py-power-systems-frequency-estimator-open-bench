from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypedDict

import matplotlib.pyplot as plt
import numpy as np

from estimators.basic.zcd import ZCDSingle
from evaluation import metrics
from evaluation.plotting import plot_signal_and_estimators
from scenarios.s1_synthetic.frequency_step import frequency_step
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output


class Estimator(Protocol):
    """Estimator interface needed here."""

    def update(self, measures: PMU_Input) -> PMU_Output:  # match concrete ZCDSingle
        ...


class RunResult(TypedDict):
    name: str
    n_samples: int
    rmse: float
    f_hat: list[float]


def run_single(
    estimator: Estimator,
    signal: np.ndarray,
    truth: np.ndarray,
    fs: float,
    channel: str,
    name: str = "unknown",
) -> RunResult:
    """Feed timestamped PMU-style measurements to the estimator and collect a frequency trace."""
    f_hat_vals: list[float | None] = []

    inv_fs = 1.0 / float(fs)
    t = 0.0
    for x in signal.tolist():
        # PMU_Input constructor expects all channels + timestamp; fill others as 0.0
        kwargs = {"V1": 0.0, "V2": 0.0, "V3": 0.0, "I1": 0.0, "I2": 0.0, "I3": 0.0, "timestamp": t}
        kwargs[channel] = float(x)
        meas = PMU_Input(**kwargs)
        out = estimator.update(meas)
        f_hat_vals.append(out.frequency_hz)
        t += inv_fs

    f_hat_arr = np.array([np.nan if v is None else float(v) for v in f_hat_vals], dtype=float)
    truth = truth[: f_hat_arr.shape[0]]

    mask = ~np.isnan(f_hat_arr)
    rmse = float(metrics.frequency_error(f_hat_arr[mask], truth[mask]))

    return {
        "name": name,
        "n_samples": int(signal.shape[0]),
        "rmse": rmse,
        "f_hat": f_hat_arr.tolist(),
    }


def main() -> None:
    fs: int = 5000
    channel = "V1"

    # ZCDSingle config
    zcd_config = {
        "epsilon": 0.0,
        "nominal_hz": 60.0,
        "mode": "neg_to_pos",
        "channel": channel,
    }
    estimators: dict[str, Estimator] = {
        "ZCD": ZCDSingle(config=zcd_config),
    }

    scenarios: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
        "s0_step": lambda: frequency_step(
            f0=60.0,
            f_step=59.5,
            t_step=1.0,
            t_back=2.0,
            duration=4.0,
            fs=fs,
        )
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path("data/results") / f"benchmark_{timestamp}"
    json_dir = root_dir / "jsons"
    plot_dir = root_dir / "plots"
    json_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[RunResult]] = {}

    for s_name, scenario_fn in scenarios.items():
        print(f"▶ Running scenario: {s_name}")
        signal, f_true = scenario_fn()

        signal = np.asarray(signal, dtype=float).ravel()
        f_true = np.asarray(f_true, dtype=float).ravel()

        results: list[RunResult] = []
        estimates: dict[str, list[float]] = {}

        for name, est in estimators.items():
            res = run_single(est, signal, f_true, fs=fs, channel=channel, name=name)
            results.append(res)
            estimates[name] = res["f_hat"]

        json_file = json_dir / f"{s_name}.json"
        with json_file.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"✅ JSON saved to {json_file}")

        fig = plot_signal_and_estimators(
            signal,
            f_true,
            estimates,
            fs,
            zoom_windows_top=[(0.95, 1.05), (1.95, 2.05)],
            zoom_window_bottom=(0.5, 2.5),
        )
        plot_file = plot_dir / f"{s_name}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Plot saved to {plot_file}")

        all_results[s_name] = results

    index_file = root_dir / "summary.json"
    with index_file.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"🗂 Summary saved to {index_file}")


if __name__ == "__main__":
    main()
