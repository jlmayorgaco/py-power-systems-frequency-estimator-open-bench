"""
openfreqbench/scenarios/g3/e20_unbalanced_sags.py

Asymmetrical Voltage Sag (Type C / Type D) Scenarios.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class UnbalancedSagScenario(ScenarioBase):
    scenario_id: ClassVar[str] = "UnbalancedSag"
    fs_hz: float = 10000.0
    duration_s: float = 5.0
    seed: int = 42
    tuning_map: ClassVar[dict[str, str]] = {"seed": "seed"}

    def build(self) -> ScenarioOutput:
        n = int(self.fs_hz * self.duration_s)
        t = np.arange(n) / self.fs_hz
        
        f_true = np.ones(n) * 60.0
        w = 2.0 * math.pi * 60.0
        
        phi_a = w * t
        phi_b = w * t - (2 * math.pi / 3)
        phi_c = w * t + (2 * math.pi / 3)
        
        mag_a = np.ones(n)
        mag_b = np.ones(n)
        mag_c = np.ones(n)
        
        sag_mask = (t >= 0.5) & (t < 1.0)
        
        d_sag = 0.5 
        mag_b[sag_mask] = d_sag
        mag_c[sag_mask] = d_sag
        phi_b[sag_mask] += math.radians(15)
        phi_c[sag_mask] -= math.radians(15)

        v_a = mag_a * np.cos(phi_a)
        v_b = mag_b * np.cos(phi_b)
        v_c = mag_c * np.cos(phi_c)
        
        v_3ph = np.stack((v_a, v_b, v_c), axis=-1)
        
        rng = np.random.RandomState(self.seed)
        v_3ph += rng.normal(0, 0.005, v_3ph.shape)

        rocof = np.zeros_like(t)
        
        state = ScenarioState(
            t=t, fs_hz=self.fs_hz, f_nom_hz=60.0, seed=self.seed,
            f_true=f_true, phi=phi_a, A=mag_a, v=v_3ph,
            roco_f_true=rocof, schema={"scenario_id": self.scenario_id, "seed": self.seed}
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
