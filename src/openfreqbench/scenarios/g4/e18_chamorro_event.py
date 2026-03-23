"""
openfreqbench/scenarios/g4/e18_chamorro_event.py

Loads realistic OpenDSS multi-event dataset from 'chamorro_case7_3ph.csv'.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

from openfreqbench.scenarios.adapters.opendss_scenario import OpenDSSPlaybackScenario
from openfreqbench.scenarios._base import ScenarioOutput

@dataclass
class ChamorroFaultScenario(OpenDSSPlaybackScenario):
    """
    Realistic 3-phase unbalance and islanding scenario derived from EPRI 
    OpenDSS simulated benchmarks (Chamorro Case 7).
    """
    csv_path: str = "legacy_sgsma/chamorro_case7_3ph.csv"
    fs_hz: float = 10000.0
    v_column_idx: int = 1  # Assuming Phase A is column 1 (ignoring time)

    scenario_id: ClassVar[str] = "G4_E18_Chamorro_Event"

    def build(self) -> ScenarioOutput:
        if not os.path.exists(self.csv_path):
            # Graceful fallback if dataset is missing local caching
            raise FileNotFoundError(f"Missing massive IEEE dataset at {self.csv_path}. Download from repo artifacts.")
        # Super build via canonical OpenDSS Playback
        return super().build()
