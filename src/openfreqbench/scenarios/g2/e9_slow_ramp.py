"""
openfreqbench/scenarios/g2/e9_slow_ramp.py

G2_E9_Slow_Ramp_minus0p5Hzs — slow frequency ramp at −0.5 Hz/s.

Signal model:
    f(t) = f_start                                      for t < t_ramp_start
           f_start + rocof_hz_s * (t - t_ramp_start)    for t_ramp_start <= t < t_ramp_end
           f_start + rocof_hz_s * (t_ramp_end - t_ramp_start)  for t >= t_ramp_end

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A0 * sin(phi(t))

RoCoF = −0.5 Hz/s is representative of a slow frequency decline typical of
primary frequency response following a minor generation deficit.  The long
ramp window (0.5 s to 3.5 s) ensures estimators must track a smoothly
varying frequency over an extended interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E9_Slow_Ramp_minus0p5Hzs(ScenarioBase):
    """
    Frequency ramp at −0.5 Hz/s from t_ramp_start to t_ramp_end, held flat
    at the final frequency thereafter.

    rocof_hz_s = −0.5 Hz/s represents a slow, benign frequency decline within
    normal grid operation.  The extended ramp period tests estimator dynamic
    tracking bias under gradual frequency evolution.  Phase continuity is
    maintained throughout the record.
    """

    fs_hz:         float = 10_000.0
    T_s:           float = 5.0
    seed:          int   = 42
    f_start:       float = 60.0
    rocof_hz_s:    float = -0.5
    t_ramp_start:  float = 0.5
    t_ramp_end:    float = 3.5
    scenario_id:   str   = "G2_E9_Slow_Ramp_minus0p5Hzs"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed": "seed",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
