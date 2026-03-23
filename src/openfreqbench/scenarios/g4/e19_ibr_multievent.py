"""
openfreqbench/scenarios/g4/e19_ibr_multievent.py

IBR MultiEvent: Composite Scenario (Phase Jumps, Fast ROCOF, Ring-down)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class IbrMultiEventScenario(ScenarioBase):
    scenario_id: ClassVar[str] = "IBR_MultiEvent"
    fs_hz: float = 10000.0
    duration_s: float = 5.0
    seed: int = 42
    tuning_map: ClassVar[dict[str, str]] = {"seed": "seed"}

    def build(self) -> ScenarioOutput:
        n = int(self.fs_hz * self.duration_s)
        t = np.arange(n) / self.fs_hz
        
        f_e = np.ones(n) * 60.0

        seg3 = (t >= 1.2) & (t < 2.5)
        seg4 = (t >= 2.5) & (t < 3.5)
        seg5 = (t >= 3.5)

        f_e[seg3] = 60.0 - 6.0 * (t[seg3] - 1.2)

        tau = t[seg4] - 2.5
        f_e[seg4] = 60.0 + 2.0 * np.exp(-3.0 * tau) * np.sin(2.0 * math.pi * 3.0 * tau)
        
        phi_e = np.zeros(n)
        curr_phi = 0.0
        for i in range(1, n):
            if 0.9999 < t[i] < 1.0001:
                curr_phi += math.radians(40.0)
            if 2.4999 < t[i] < 2.5001:
                curr_phi += math.radians(80.0)
            curr_phi += 2.0 * math.pi * f_e[i] * (1.0 / self.fs_hz)
            phi_e[i] = curr_phi

        amp_e = np.ones(n)
        amp_e[seg3] = 1.0 - 0.15 * (t[seg3] - 1.2) / (2.5 - 1.2)
        amp_e[seg4] = 0.85 + 0.2 * np.exp(-3.0 * tau) * np.cos(2.0 * math.pi * 3.0 * tau)
        amp_e[seg5] = 1.05

        v_base = amp_e * np.sin(phi_e)
        v_base += 0.05 * np.sin(5 * phi_e)
        v_base += 0.03 * np.sin(7 * phi_e)

        rng = np.random.RandomState(self.seed)
        v_e = v_base + rng.normal(0, 0.003, n)
        impulse_mask = (rng.rand(n) < 5e-4)
        v_e += impulse_mask * rng.normal(0, 0.05, n)

        rocof = np.zeros_like(t)

        state = ScenarioState(
            t=t, fs_hz=self.fs_hz, f_nom_hz=60.0, seed=self.seed,
            f_true=f_e, phi=phi_e, A=amp_e, v=v_e.reshape(-1, 1),
            roco_f_true=rocof, schema={"scenario_id": self.scenario_id, "seed": self.seed}
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
