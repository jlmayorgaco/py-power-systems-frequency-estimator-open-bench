"""
openfreqbench/scenarios/g4/e19_ibr_multievent.py

IBR MultiEvent: Composite Scenario (Phase Jumps, Fast ROCOF, Ring-down)
Ported from legacy_sgsma/scenarios.py
"""

from __future__ import annotations

import math
import numpy as np

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput

class IbrMultiEventScenario(ScenarioBase):
    def build(self) -> ScenarioOutput:
        n = int(self.fs_hz * 5.0)
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

        v_e = v_base + np.random.normal(0, 0.003, n)
        impulse_mask = (np.random.rand(n) < 5e-4)
        v_e += impulse_mask * np.random.normal(0, 0.05, n)

        rocof = np.zeros_like(t)

        return ScenarioOutput(
            time_s=t,
            voltage_pu=v_e,
            frequency_hz=f_e,
            rocof_hz_s=rocof,
            metadata={"description": "Composite sequence of severe active islanding."}
        )
