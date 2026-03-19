"""
openfreqbench/scenarios/g2/e8_fast_ramp.py

G2_E8_Fast_Ramp_plus5Hzs — fast frequency ramp at +5 Hz/s.

Signal model:
    f(t) = f_start                                      for t < t_ramp_start
           f_start + rocof_hz_s * (t - t_ramp_start)    for t_ramp_start <= t < t_ramp_end
           f_start + rocof_hz_s * (t_ramp_end - t_ramp_start)  for t >= t_ramp_end

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A0 * sin(phi(t))

RoCoF = +5 Hz/s is representative of fast frequency excursions following
sudden loss of generation or load, and exceeds typical RoCoF relay trip
thresholds (2 Hz/s in many jurisdictions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E8_Fast_Ramp_plus5Hzs(ScenarioBase):
    """
    Frequency ramp at +5 Hz/s from t_ramp_start to t_ramp_end, held flat
    at the final value thereafter.

    rocof_hz_s = +5 Hz/s is a fast ramp that exceeds standard RoCoF protection
    thresholds.  The pre-ramp and post-ramp segments at constant frequency
    allow measurement of estimator steady-state bias before and after the event.
    Phase continuity is maintained throughout.
    """

    fs_hz:         float = 10_000.0
    T_s:           float = 3.0
    seed:          int   = 42
    f_start:       float = 60.0
    rocof_hz_s:    float = 5.0
    t_ramp_start:  float = 0.5
    t_ramp_end:    float = 1.5
    scenario_id:   str   = "G2_E8_Fast_Ramp_plus5Hzs"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed": "seed",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
