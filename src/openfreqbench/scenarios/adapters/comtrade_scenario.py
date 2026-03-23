"""
openfreqbench/scenarios/adapters/comtrade_scenario.py

Adapter to parse COMTRADE files into canonical ScenarioBase representations.
"""

from __future__ import annotations

import numpy as np
from typing import ClassVar
from dataclasses import dataclass

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState

@dataclass
class COMTRADEScenario(ScenarioBase):
    """
    Parses IEEE COMTRADE (.cfg / .dat) files as a benchmark scenario.
    """
    cfg_path: str = ""
    dat_path: str = ""
    channel_idx: int = 0
    
    scenario_id: ClassVar[str] = "COMTRADE_Playback"
    
    def build(self) -> ScenarioOutput:
        if not self.cfg_path or not self.dat_path:
            raise ValueError("cfg_path and dat_path must be specified.")
            
        try:
            import comtrade
        except ImportError:
            raise ImportError("Please install 'comtrade' (pip install comtrade) to use COMTRADEScenario.")

        rec = comtrade.Comtrade()
        rec.load(self.cfg_path, self.dat_path)
        
        fs = rec.cfg.sample_rates[0][0]
        v_data = np.array(rec.analog[self.channel_idx], dtype=float)
        
        # Determine nominal manually if possible, or assume 60Hz
        f_nom = 60.0
        
        state = ScenarioState(
            fs_hz=fs,
            f_nom_hz=f_nom,
            t=np.arange(len(v_data)) / fs,
            v=v_data.reshape(-1, 1),
            f_true=np.full(len(v_data), float("nan")), # Unknown true freq in playback
        )
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
