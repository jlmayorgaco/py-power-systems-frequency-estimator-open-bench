from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E11_FM_Modulation(ScenarioBase):
    """
    Scenario: Frequency Modulation (FM).

    Signal Model:
      f(t) = f_nom + k_m * cos(2*pi * f_m * t)
      v(t) = A0 * sin(phi(t))

      where:
        k_m = mod_depth_hz (Peak frequency deviation, e.g., 2.0 Hz)
        f_m = mod_freq_hz (Modulation Frequency)

    Phase is calculated via integration of f(t) to ensure physical consistency.
    Standard IEEE C37.118 test for bandwidth (C37.118.1a-2014).
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # FM Parameters
    mod_freq_hz: float = 2.0  # f_m (Modulation Speed)
    mod_depth_hz: float = 2.0  # k_m (Frequency Deviation, e.g. +/- 2Hz)

    scenario_id: str = "G3_E11_FM_Modulation"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # FM Specific
            "fm_freq": "mod_freq_hz",  # Speed of oscillation (Hz)
            "fm_depth": "mod_depth_hz",  # Deviation Magnitude (Hz)
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        # 1. Time base
        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # 2. Physics: Frequency Modulation
        # f(t) = f0 + k * cos(wm * t)

        f_m = float(self.mod_freq_hz)
        k_m = float(self.mod_depth_hz)

        # Using Cosine convention for modulation peak at t=0
        oscillation = np.cos(2.0 * np.pi * f_m * t)
        f_true = float(self.f_nom_hz) + (k_m * oscillation)

        # 3. Physics: Amplitude (Constant)
        A = np.full_like(t, float(self.A0), dtype=float)

        # 4. Initialize State
        # Phase and Voltage will be computed via Integration Mixin
        dummy_zeros = np.zeros_like(t)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=dummy_zeros,
            A=A,
            v=dummy_zeros,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "signal_type": "FM",
                "mod_freq_hz": f_m,
                "mod_depth_hz": k_m,
                "modifiers": [],
            },
        )

        self.state = state

        # --- DRY: Compute Physics (Integral) ---
        # phi = Integral(f_true)
        # This handles the complex FM phase equation automatically
        self.recompute_phi_from_f(phi0_rad=float(self.phi0_rad))
        self.recompute_v_from_A_phi()

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
