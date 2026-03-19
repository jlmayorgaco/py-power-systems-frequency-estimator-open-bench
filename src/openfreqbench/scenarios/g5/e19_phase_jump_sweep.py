"""
openfreqbench/scenarios/g5/e19_phase_jump_sweep.py

G5_E19_Phase_Jump_Sweep — parametric sweep over phase-jump magnitude.

Signal model (identical to G3_E12_Phase_Jump):
    phi(t) = 2*pi*f_nom*t                       for t < t_jump
             2*pi*f_nom*t + jump_rad              for t >= t_jump

    v(t)   = A0 * sin(phi(t))

The distinguishing feature of this scenario is its Monte Carlo tuning design.
The tuning_map routes the sweep alias "jump_rad" → sweep_value, so the MC
harness injects different jump magnitudes into sweep_value.  jump_rad is
then set equal to sweep_value at build time.

This allows systematic characterisation of estimator phase-step RMSE as a
function of jump magnitude by running this scenario across a parameter grid
(e.g., jump_rad ∈ [0.1, 0.2, ..., 1.5] rad).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G5_E19_Phase_Jump_Sweep(ScenarioBase):
    """
    Parametric sweep scenario for phase-jump magnitude characterisation.

    sweep_value is the MC-injectable parameter (alias "jump_rad" in tuning_map).
    At build time jump_rad is set to sweep_value so that each MC trial uses
    a different phase-jump magnitude.  True frequency is constant at f_nom = 60 Hz.
    Enables RMSE-vs-jump-magnitude curves across all estimators in one run.
    """

    fs_hz:       float = 10_000.0
    T_s:         float = 2.0
    seed:        int   = 42
    f_nom:       float = 60.0
    t_jump:      float = 0.5
    jump_rad:    float = 0.5236   # pi/6 ≈ 30 degrees
    sweep_value: float = 0.5236   # set by MC tuning via tuning_map
    scenario_id: str   = "G5_E19_Phase_Jump_Sweep"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":     "seed",
        "jump_rad": "sweep_value",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
