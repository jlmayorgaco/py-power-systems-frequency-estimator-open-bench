"""
openfreqbench/scenarios/g3/e12_phase_jump.py

G3_E12_Phase_Jump — instantaneous phase discontinuity at t_jump.

Signal model:
    phi(t) = 2*pi*f_nom*t                   for t < t_jump
             2*pi*f_nom*t + jump_rad          for t >= t_jump

    v(t)   = A0 * sin(phi(t))

The true instantaneous frequency is constant at f_nom = 60 Hz throughout;
only the absolute phase shifts by jump_rad at t_jump.

Default: jump_rad = π/6 ≈ 0.5236 rad (30°).

Phase jumps arise from tap-changer operations, capacitor switching, and
fault clearance.  Estimators that track phase via wrapped integration will
exhibit a transient frequency spike of finite duration at the jump instant.
"""

from __future__ import annotations

from typing import ClassVar

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E12_Phase_Jump(ScenarioBase):
    """
    Instantaneous phase jump of jump_rad at t_jump seconds.

    True frequency remains constant at f_nom = 60 Hz.  The phase
    discontinuity (default 30° = π/6 rad) is introduced by directly
    offsetting the accumulated phase angle at t_jump.  Tests estimator
    handling of sudden phase changes without true frequency deviation.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    f_nom: float = 60.0
    t_jump: float = 0.5
    jump_rad: float = 0.5236  # pi/6 ≈ 30 degrees
    scenario_id: ClassVar[str] = "G3_E12_Phase_Jump"

    tuning_map: ClassVar[dict[str, str]] = {
            "seed": "seed",
            "t_jump": "t_jump",
            "jump_rad": "jump_rad",
        }

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
