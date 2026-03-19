from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G1_E1_Pure_60Hz(ScenarioBase):
    """
    Scenario: Pure nominal sine wave at fixed frequency (no events, no noise).
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 0

    scenario_id: str = "G1_E1_Pure_60Hz"

    # --- Monte Carlo Adapter Map ---
    # Maps 'Domain Alias' -> 'Internal Attribute'
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",  # Target frequency
            "freq": "f_nom_hz",  # Alternative alias
            "Vmax": "A0",  # Amplitude
            "phi": "phi0_rad",  # Phase offset
            "seed": "seed",  # Random seed
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

        # 2. Physics (Frequency & Phase)
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        # phi(t) = phi0 + 2π * f * t
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        # 3. Voltage Signal
        A = np.full_like(t, float(self.A0), dtype=float)
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
                "T_s": T,
                "A0": float(self.A0),
                "phi0_rad": float(self.phi0_rad),
                "modifiers": [],
            },
        )

        # Assign state to self so Mixin methods can access it if needed later
        self.state = state

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
