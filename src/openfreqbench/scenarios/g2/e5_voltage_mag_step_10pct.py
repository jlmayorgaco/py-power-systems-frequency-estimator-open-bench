"""
openfreqbench/scenarios/g2/e5_voltage_mag_step_10pct.py

G2_E5_Voltage_Mag_Step_10pct — abrupt 10% voltage magnitude step.

Signal model:
    A(t) = 1.0             for t < t_step
           1.0 + delta_pu  for t >= t_step   (delta_pu = 0.10)

    phi(t) = 2*pi*60*t   (constant frequency, phase-continuous)
    v(t)   = A(t) * sin(phi(t))

True frequency is constant at 60 Hz throughout; only amplitude changes.
The larger step (10x) compared to E4 exposes estimators that conflate
instantaneous amplitude and frequency (e.g., envelope-based methods).
"""

from __future__ import annotations

from typing import ClassVar

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E5_Voltage_Mag_Step_10pct(ScenarioBase):
    """
    Pure 60 Hz sine with an abrupt 10% voltage magnitude step at t_step seconds.

    Amplitude transitions from 1.0 pu to 1.10 pu (delta_pu = 0.10) at t_step = 0.5 s.
    Frequency truth remains constant at 60 Hz.  The large amplitude transient
    is a known excitation source for false frequency deviations in zero-crossing
    and analytic-signal based estimators.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    t_step: float = 0.5
    delta_pu: float = 0.10
    scenario_id: ClassVar[str] = "G2_E5_Voltage_Mag_Step_10pct"

    tuning_map: ClassVar[dict[str, str]] = {
            "seed": "seed",
            "t_step": "t_step",
        }

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
