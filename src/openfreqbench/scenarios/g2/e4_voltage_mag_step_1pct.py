"""
openfreqbench/scenarios/g2/e4_voltage_mag_step_1pct.py

G2_E4_Voltage_Mag_Step_1pct — abrupt 1% voltage magnitude step.

Signal model:
    A(t) = 1.0           for t < t_step
           1.0 + delta_pu  for t >= t_step   (delta_pu = 0.01)

    phi(t) = 2*pi*60*t   (constant frequency, phase-continuous)
    v(t)   = A(t) * sin(phi(t))

True frequency is constant at 60 Hz throughout; only amplitude changes.
This scenario tests estimator immunity to abrupt amplitude disturbances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E4_Voltage_Mag_Step_1pct(ScenarioBase):
    """
    Pure 60 Hz sine with an abrupt 1% voltage magnitude step at t_step seconds.

    Amplitude transitions from 1.0 pu to 1.01 pu (delta_pu = 0.01) at t_step = 0.5 s.
    Frequency truth remains constant at 60 Hz.  Tests estimator sensitivity to
    amplitude disturbances that should not affect frequency estimates.
    """

    fs_hz:       float = 10_000.0
    T_s:         float = 2.0
    seed:        int   = 42
    t_step:      float = 0.5
    delta_pu:    float = 0.01
    scenario_id: str   = "G2_E4_Voltage_Mag_Step_1pct"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":   "seed",
        "t_step": "t_step",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
