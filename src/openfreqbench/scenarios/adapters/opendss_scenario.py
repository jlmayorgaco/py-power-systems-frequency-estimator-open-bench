"""
openfreqbench/scenarios/adapters/opendss_scenario.py

Adapter to load OpenDSS CSV playback files into ScenarioBase representations.
"""

from __future__ import annotations

import numpy as np
from typing import ClassVar
from dataclasses import dataclass

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class OpenDSSPlaybackScenario(ScenarioBase):
    """
    Reads OpenDSS exported CSV curves for algorithm playback testing.
    """
    csv_path: str = ""
    fs_hz: float = 10000.0
    v_column_idx: int = 1
    
    scenario_id: ClassVar[str] = "OpenDSS_Playback"
    
    def build(self) -> ScenarioOutput:
        if not self.csv_path:
            raise ValueError("csv_path must be provided.")
            
        data = np.genfromtxt(self.csv_path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            v_data = data
        else:
            v_data = data[:, self.v_column_idx]
            
        f_nom = 60.0
        
        state = ScenarioState(
            fs_hz=self.fs_hz,
            f_nom_hz=f_nom,
            t=np.arange(len(v_data)) / self.fs_hz,
            v=v_data.reshape(-1, 1),
            f_true=np.full(len(v_data), float("nan")),
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
