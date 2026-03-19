from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G4_E18_Chamorro_Event(ScenarioBase):
    """
    Scenario: Real Event Playback (Chamorro Event).

    Loads a CSV file containing real disturbance recorder data:
      Columns expected: t, f, va, vb, vc

    Features:
      - Automatic Resampling: Interpolates CSV data to match the requested 'fs_hz'.
      - Ground Truth Alignment: Uses the CSV 'f' column as f_true.
      - Phase A Extraction: Maps 'va' to the simulation's primary voltage 'v'.
    """

    # Simulation settings
    fs_hz: float = 10_000.0  # Target sampling rate

    # File settings
    csv_path: str = "data/chamorro_event.csv"  # Path to your file

    # Column Mapping (Adjust if your CSV headers differ)
    col_t: str = "t"
    col_f: str = "f"
    col_va: str = "va"
    col_vb: str = "vb"  # Optional, stored in metadata
    col_vc: str = "vc"  # Optional, stored in metadata

    # Playback Tuning (Monte Carlo)
    time_shift_s: float = 0.0
    amplitude_scale: float = 1.0

    scenario_id: str = "G4_E18_Chamorro_Event"

    # --- Monte Carlo Adapter ---
    # Allows shifting the event or scaling it randomly
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "shift": "time_shift_s",
            "scale": "amplitude_scale",
            "file": "csv_path",
        }
    )

    def build(self) -> ScenarioOutput:
        # 1. Load Data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Check columns
        required_cols = [self.col_t, self.col_f, self.col_va]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(
                    f"Missing required column '{c}' in CSV. Found: {df.columns}"
                )

        # Extract raw arrays
        t_raw = df[self.col_t].values
        f_raw = df[self.col_f].values
        va_raw = df[self.col_va].values

        # 2. Time Alignment & Resampling
        # We create a new time vector based on fs_hz and the duration of the CSV
        t_start = t_raw[0]
        t_end = t_raw[-1]
        duration = t_end - t_start

        # Number of samples for the target fs
        n_samples = int(duration * self.fs_hz)
        t_new = np.linspace(t_start, t_end, n_samples)

        # Apply Time Shift (if requested via Monte Carlo)
        # Shift only shifts the "window", effectively shifting the data in time
        t_new_shifted = t_new - self.time_shift_s

        # Interpolators (Linear is usually fine for dense data, Cubic for sparse)
        # fill_value="extrapolate" handles minor edge cases due to shifting
        interp_f = interp1d(t_raw, f_raw, kind="linear", fill_value="extrapolate")
        interp_va = interp1d(t_raw, va_raw, kind="cubic", fill_value="extrapolate")

        # 3. Generate Resampled Signals
        f_true = interp_f(t_new_shifted)
        v = interp_va(t_new_shifted) * float(self.amplitude_scale)

        # Optional: Phase B and C (Just strictly reading, not putting in main state for now)
        # If we wanted full 3-phase state, we'd expand ScenarioState

        # 4. Physics: Phase Reconstruction
        # Real data usually has noisy 'f'. Integrating it gives 'phi'.
        # However, to be consistent with 'va', we might not want to synthesize phi
        # unless strictly needed. But ScenarioState expects it.
        # Let's integrate f_true to get a reference phase.

        # phi(t) = 2*pi * integral(f)
        # Cumsum approximation
        dt = 1.0 / self.fs_hz
        phi = 2.0 * np.pi * np.cumsum(f_true) * dt
        # Normalize to start at 0 (or keep raw integral)
        phi -= phi[0]

        # 5. Amplitude Estimation (Optional)
        # Since v = A * sin(phi), we don't have explicit A from CSV (unless computed).
        # We put a dummy A=1.0 or compute Envelope (Hilbert).
        # Let's stick to dummy A=1.0 to indicate "unknown/variable".
        A_dummy = np.ones_like(t_new)

        state = ScenarioState(
            t=t_new,
            fs_hz=float(self.fs_hz),
            f_nom_hz=60.0,  # Assumed nominal, or infer from mean(f_raw)
            seed=0,  # Deterministic playback
            f_true=f_true,
            phi=phi,
            A=A_dummy,
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "source_file": self.csv_path,
                "duration_s": duration,
                "resampled_fs": float(self.fs_hz),
                "time_shift": float(self.time_shift_s),
                "modifiers": ["playback"],
            },
        )
        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
