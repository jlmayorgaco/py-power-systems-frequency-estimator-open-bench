from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E10_AM_Modulation(ScenarioBase):
    """
    Scenario: Amplitude Modulation (AM).

    Signal Model:
      v(t) = A(t) * sin(phi(t))

      A(t) = A0 * [1 + k_m * cos(2*pi * f_m * t)]

      where:
        k_m = mod_depth (e.g., 0.1 for 10%)
        f_m = mod_freq_hz (Modulation Frequency)

    Standard IEEE C37.118 test for bandwidth and interference rejection.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # AM Parameters
    mod_freq_hz: float = 2.0  # f_m (Typically 0.1 to 5.0 Hz)
    mod_depth: float = 0.1  # k_m (Typically 0.1 or 10%)

    scenario_id: str = "G3_E10_AM_Modulation"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # AM Specific
            "am_freq": "mod_freq_hz",  # Frequency of the oscillation
            "am_depth": "mod_depth",  # Magnitude of the oscillation
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

        # 2. Physics: Constant Frequency (Carrier)
        # In pure AM, the instantaneous frequency of the carrier is constant.
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        # Carrier Phase
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        # 3. Physics: Amplitude Modulation (The Envelope)
        # A(t) = A0 * (1 + k_m * cos(omega_m * t))
        # Using Cosine is standard convention so peak is at t=0

        k_m = float(self.mod_depth)
        f_m = float(self.mod_freq_hz)

        mod_signal = np.cos(2.0 * np.pi * f_m * t)
        A = float(self.A0) * (1.0 + (k_m * mod_signal))

        # 4. Final Voltage
        v = A * np.sin(phi)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=phi,
            A=A,
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "signal_type": "AM",
                "mod_freq_hz": f_m,
                "mod_depth": k_m,
                "modifiers": [],
            },
        )

        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
