"""
openfreqbench/scenarios/g2/e21_oobi_interference.py

Out-of-Band Interference (OOBI) scenario.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class OOBIInterferenceScenario(ScenarioBase):
    scenario_id: ClassVar[str] = "OOBIInterference"
    fs_hz: float = 10000.0
    duration_s: float = 5.0
    seed: int = 42
    tuning_map: ClassVar[dict[str, str]] = {"seed": "seed"}

    def build(self) -> ScenarioOutput:
        n = int(self.fs_hz * self.duration_s)
        t = np.arange(n) / self.fs_hz
        
        f_true = np.ones(n) * 60.0
        w = 2.0 * math.pi * 60.0
        
        phi = w * t
        v_fund = np.sin(phi)
        
        f_inter = np.linspace(10.0, 110.0, n)
        phi_inter = 2.0 * math.pi * np.cumsum(f_inter) / self.fs_hz
        
        v_oobi = 0.1 * np.sin(phi_inter)
        v_total = v_fund + v_oobi
        
        rocof = np.zeros_like(t)

        state = ScenarioState(
            t=t, fs_hz=self.fs_hz, f_nom_hz=60.0, seed=self.seed,
            f_true=f_true, phi=phi, A=np.ones(n), v=v_total.reshape(-1, 1),
            roco_f_true=rocof, schema={"scenario_id": self.scenario_id, "seed": self.seed}
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
