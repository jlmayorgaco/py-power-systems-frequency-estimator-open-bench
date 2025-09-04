# pipelines/run_benchmark.py

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from evaluation import metrics
from evaluation.plotting import plot_signal_and_estimators

from estimators.basic.ipdft import IpDFT
from estimators.basic.zcd import ZeroCrossing

from scenarios.s1_synthetic.make_clean import make_clean
from scenarios.s1_synthetic.frequency_step import frequency_step
from scenarios.s1_synthetic.frequency_ramp_step import frequency_ramp_step

# Later you can add more scenarios (IEEE 13-bus, noisy, etc.)


def run_single(estimator, signal, truth, name="unknown"):
    """Run estimator online: feed one sample at a time, get frequency trace."""
    f_hat = []
    for st in signal:
        val = estimator.update(st)
        f_hat.append(val)

    # Convert to array, replace None with NaN
    f_hat = np.array([np.nan if v is None else v for v in f_hat])
    truth = truth[:len(f_hat)]

    # Compute RMSE ignoring NaNs
    mask = ~np.isnan(f_hat)
    rmse = metrics.frequency_error(f_hat[mask], truth[mask])

    return {
        "name": name,
        "n_samples": len(signal),
        "rmse": rmse,
        "f_hat": f_hat.tolist(),
    }




def main():
    fs = 5000

    # === Define estimators ===
    estimators = {
        "ZCD": ZeroCrossing(fs=fs)
        #"IpDFT": IpDFT(fs=fs, frame_len=256),
    }

    # === Define scenarios ===
    scenarios = {

        # Step change: 60 Hz → 59.5 Hz → back to 60 Hz
        "s0_step": lambda: frequency_step(
            f0=60.0, f_step=50, t_step=1.0, t_back=2.0, duration=4.0, fs=fs
        )
    }

    # === Timestamped results folder ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_DIR = pathlib.Path("data/results") / f"benchmark_{timestamp}"
    JSON_DIR = ROOT_DIR / "jsons"
    PLOT_DIR = ROOT_DIR / "plots"
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # === Run each scenario ===
    for s_name, scenario_fn in scenarios.items():
        print(f"▶ Running scenario: {s_name}")
        signal, f_true = scenario_fn()

        results = []
        estimates = {}

        # Run all estimators
        for name, est in estimators.items():
            res = run_single(est, signal, f_true, name=name)
            results.append(res)
            estimates[name] = res["f_hat"]

        # Save JSON
        json_file = JSON_DIR / f"{s_name}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ JSON saved to {json_file}")

        # Save Plot
        fig = plot_signal_and_estimators(
            signal, f_true, estimates, fs,
            zoom_windows_top=[(0.95, 1.05), (1.95, 2.05)],   # show two windows on the raw signal
            zoom_window_bottom=(0.5, 2.5)                 # zoom for frequency subplot
        )

        plot_file = PLOT_DIR / f"{s_name}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Plot saved to {plot_file}")

        all_results[s_name] = results

    # Optionally: save a global index of all scenarios
    index_file = ROOT_DIR / "summary.json"
    with open(index_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"🗂 Summary saved to {index_file}")


if __name__ == "__main__":
    main()
