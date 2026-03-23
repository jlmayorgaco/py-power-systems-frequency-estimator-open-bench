"""
openfreqbench/scenarios/g2/e15_ibr_nightmare.py

IBR_Nightmare: Phase Jump (+60 deg) + 5th Harmonic + Inter-harmonic
Ported from legacy_sgsma/scenarios.py
"""

from __future__ import annotations

import math
import numpy as np

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput

class IbrNightmareScenario(ScenarioBase):
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
        v_d += np.random.normal(0, 0.005, n)            

        rocof = np.zeros_like(t)
        
        return ScenarioOutput(
            time_s=t,
            voltage_pu=v_d,
            frequency_hz=f_d,
            rocof_hz_s=rocof,
            metadata={"description": "Instantaneous Phase Jump +60 deg", "harmonics": "5th (5%), Inter-harmonic 32.5Hz (2%)"}
        )
