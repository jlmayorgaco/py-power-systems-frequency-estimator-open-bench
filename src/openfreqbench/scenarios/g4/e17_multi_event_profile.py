"""
openfreqbench/scenarios/g4/e17_multi_event_profile.py

G4_E17_Multi_Event_Profile — multi-event sequence testing estimator state persistence.

Event sequence (default t_events = [0.5, 1.5, 2.5]):
    t < 0.5 s  : steady-state 60 Hz, unit amplitude, low noise
    t = 0.5 s  : Event 1 — abrupt voltage magnitude step (+5%)
    t = 1.5 s  : Event 2 — abrupt frequency step (60 → 60.5 Hz)
    t = 2.5 s  : Event 3 — noise level increase (SNR drops by 20 dB)
    t > 2.5 s  : sustained modified conditions until T_s = 4.0 s

The three-event chain exercises whether estimator internal state (filter
memory, PLL integrators, etc.) carries artefacts from earlier events into
the assessment window of later events — a failure mode not revealed by
single-event scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G4_E17_Multi_Event_Profile(ScenarioBase):
    """
    Sequential three-event profile: voltage step at t=0.5 s, frequency step at
    t=1.5 s, and noise intensification at t=2.5 s.

    Each event modifies one signal property independently while the others
    persist.  The scenario reveals state-persistence failures, where a
    transient from an earlier event contaminates estimation during a
    later event window.  The 4-second record provides ample pre- and
    post-event analysis regions.
    """

    fs_hz: float = 10_000.0
    T_s: float = 4.0
    seed: int = 42
    t_events: list[float] = field(default_factory=lambda: [0.5, 1.5, 2.5])
    scenario_id: str = "G4_E17_Multi_Event_Profile"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "seed": "seed",
        },
    )

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
