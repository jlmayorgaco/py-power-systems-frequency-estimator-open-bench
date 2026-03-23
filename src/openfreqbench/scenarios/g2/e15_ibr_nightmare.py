"""
openfreqbench/scenarios/g2/e15_ibr_nightmare.py

IBR_Nightmare: Phase Jump (+60 deg) + 5th Harmonic + Inter-harmonic
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class IbrNightmareScenario(ScenarioBase):
    scenario_id: ClassVar[str] = "IBR_Nightmare"
    fs_hz: float = 10000.0
    duration_s: float = 5.0
    seed: int = 42
    tuning_map: ClassVar[dict[str, str]] = {"seed": "seed"}

    def build(self) -> ScenarioOutput:
        n = int(self.fs_hz * self.duration_s)
        t = np.arange(n) / self.fs_hz
        
        f_d = np.ones(n) * 60.0
        phase_accum = np.zeros(n)
        curr_phi = 0.0
        
        for i in range(n):
            if i > 0:
                if 0.6999 < t[i] < 0.7001:
                    curr_phi += math.pi / 3.0
                curr_phi += 2 * math.pi * 60.0 * (1.0 / self.fs_hz)
            phase_accum[i] = curr_phi

        v_d = np.sin(phase_accum)
        v_d += 0.05 * np.sin(5 * phase_accum)           
        v_d += 0.02 * np.sin(2 * math.pi * 32.5 * t)      
        v_d += np.random.RandomState(self.seed).normal(0, 0.005, n)            

        rocof = np.zeros_like(t)
        
        state = ScenarioState(
            t=t, fs_hz=self.fs_hz, f_nom_hz=60.0, seed=self.seed,
            f_true=f_d, phi=phase_accum, A=np.ones(n), v=v_d.reshape(-1, 1),
            roco_f_true=rocof, schema={"scenario_id": self.scenario_id, "seed": self.seed}
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
