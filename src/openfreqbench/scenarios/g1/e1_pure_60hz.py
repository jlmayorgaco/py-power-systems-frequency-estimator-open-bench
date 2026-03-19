"""
openfreqbench/scenarios/g1/e1_pure_60hz.py

G1_E1: Pure 60 Hz sine wave — no noise, no events.
Ported from pfebench/scenarios/G1_E1_Pure_60Hz.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G1_E1_Pure_60Hz(ScenarioBase):
    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 0
    scenario_id: str = "G1_E1_Pure_60Hz"
    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "freq": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
        },
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        n = max(2, round(self.T_s * fs))
        t = np.arange(n, dtype=float) / fs
        f_true = np.full(n, float(self.f_nom_hz))
        phi = float(self.phi0_rad) + 2.0 * np.pi * float(self.f_nom_hz) * t
        A = np.full(n, float(self.A0))
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
                "T_s": float(self.T_s),
                "A0": float(self.A0),
                "phi0_rad": float(self.phi0_rad),
                "modifiers": [],
            },
        )
        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
