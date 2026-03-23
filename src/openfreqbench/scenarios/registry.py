"""
openfreqbench/scenarios/registry.py

ScenarioRegistry: scenario_id -> class mapping with built-in registrations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from openfreqbench.scenarios._base import ScenarioBase
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz
from openfreqbench.scenarios.g1.e2_gaussian_noise_1pct import (
    G1_E2_Gaussian_Noise_1pct,
)
from openfreqbench.scenarios.g1.e3_gaussian_noise_5pct import (
    G1_E3_Gaussian_Noise_5pct,
)
from openfreqbench.scenarios.g2.e1_freq_step import G2_E1_FreqStep
from openfreqbench.scenarios.g2.e2_freq_ramp import G2_E2_FreqRamp
from openfreqbench.scenarios.g2.e4_voltage_mag_step_1pct import (
    G2_E4_Voltage_Mag_Step_1pct,
)
from openfreqbench.scenarios.g2.e5_voltage_mag_step_10pct import (
    G2_E5_Voltage_Mag_Step_10pct,
)
from openfreqbench.scenarios.g2.e6_freq_step_59p5 import G2_E6_Freq_Step_60_to_59p5
from openfreqbench.scenarios.g2.e7_freq_step_55 import G2_E7_Freq_Step_60_to_55
from openfreqbench.scenarios.g2.e8_fast_ramp import G2_E8_Fast_Ramp_plus5Hzs
from openfreqbench.scenarios.g2.e9_slow_ramp import G2_E9_Slow_Ramp_minus0p5Hzs
from openfreqbench.scenarios.g3.e10_am_modulation import G3_E10_AM_Modulation
from openfreqbench.scenarios.g3.e11_fm_modulation import G3_E11_FM_Modulation
from openfreqbench.scenarios.g3.e12_phase_jump import G3_E12_Phase_Jump
from openfreqbench.scenarios.g3.e13_impulsive_outliers import (
    G3_E13_Impulsive_Outliers,
)
from openfreqbench.scenarios.g3.e14_noise_harmonics import G3_E14_Noise_Harmonics
from openfreqbench.scenarios.g3.e15_noise_interharmonics import (
    G3_E15_Noise_Interharmonics,
)
from openfreqbench.scenarios.g4.e16_composite_islanding import (
    G4_E16_Composite_Islanding,
)
from openfreqbench.scenarios.g4.e17_multi_event_profile import (
    G4_E17_Multi_Event_Profile,
)
from openfreqbench.scenarios.g4.e18_chamorro_event import ChamorroFaultScenario
from openfreqbench.scenarios.g2.e15_ibr_nightmare import IbrNightmareScenario
from openfreqbench.scenarios.g4.e19_ibr_multievent import IbrMultiEventScenario
from openfreqbench.scenarios.g3.e20_unbalanced_sags import UnbalancedSagScenario
from openfreqbench.scenarios.g2.e21_oobi_interference import OOBIInterferenceScenario
from openfreqbench.scenarios.g5.e19_phase_jump_sweep import G5_E19_Phase_Jump_Sweep
from openfreqbench.scenarios.g5.e20_snr_sweep import G5_E20_SNR_Sweep


class ScenarioRegistry:
    _registry: ClassVar[dict[str, type[ScenarioBase]]] = {}
    
    # ── Methodological Isolation: Tuning vs. Testing ─────────────────────────────
    # TUNING_SCENARIOS are visible to hyperparameter optimization sweeps (GridSearch,
    # calibration) to fit standard PMU compliance characteristics (e.g., IEEE).
    # TESTING_SCENARIOS must be strictly reserved for zero-shot performance 
    # generalization assessment to prevent algorithmic data leakage.
    TUNING_SCENARIOS: ClassVar[tuple[str, ...]] = (
        "G1_E1_Pure_60Hz",
        "G1_E2_Gaussian_Noise_1pct",
        "G1_E3_Gaussian_Noise_5pct",
        "G2_E1_FreqStep",
        "G2_E2_FreqRamp",
        "G3_E10_AM_Modulation",
    )

    @classmethod
    def register(cls, scenario_cls: type[ScenarioBase]) -> type[ScenarioBase]:
        cls._registry[scenario_cls.scenario_id] = scenario_cls
        return scenario_cls

    @classmethod
    def get(cls, name: str) -> type[ScenarioBase]:
        if name not in cls._registry:
            raise KeyError(f"Scenario {name!r} not registered. Available: {sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def build(cls, name: str, params: dict[str, Any] | None = None) -> ScenarioBase:
        cls_ = cls.get(name)
        return cls_(**params) if params else cls_()

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def list_tuning_names(cls) -> list[str]:
        return [s for s in cls.TUNING_SCENARIOS if s in cls._registry]

    @classmethod
    def list_testing_names(cls) -> list[str]:
        return [s for s in cls._registry if s not in cls.TUNING_SCENARIOS]


ScenarioRegistry.register(G1_E1_Pure_60Hz)
ScenarioRegistry.register(G2_E1_FreqStep)
ScenarioRegistry.register(G2_E2_FreqRamp)

# ── Scaffold scenario registrations (build() raises NotImplementedError) ──────
ScenarioRegistry.register(G1_E2_Gaussian_Noise_1pct)
ScenarioRegistry.register(G1_E3_Gaussian_Noise_5pct)
ScenarioRegistry.register(G2_E4_Voltage_Mag_Step_1pct)
ScenarioRegistry.register(G2_E5_Voltage_Mag_Step_10pct)
ScenarioRegistry.register(G2_E6_Freq_Step_60_to_59p5)
ScenarioRegistry.register(G2_E7_Freq_Step_60_to_55)
ScenarioRegistry.register(G2_E8_Fast_Ramp_plus5Hzs)
ScenarioRegistry.register(G2_E9_Slow_Ramp_minus0p5Hzs)
ScenarioRegistry.register(G3_E10_AM_Modulation)
ScenarioRegistry.register(G3_E11_FM_Modulation)
ScenarioRegistry.register(G3_E12_Phase_Jump)
ScenarioRegistry.register(G3_E13_Impulsive_Outliers)
ScenarioRegistry.register(G3_E14_Noise_Harmonics)
ScenarioRegistry.register(G3_E15_Noise_Interharmonics)
ScenarioRegistry.register(G4_E16_Composite_Islanding)
ScenarioRegistry.register(G4_E17_Multi_Event_Profile)
ScenarioRegistry.register(ChamorroFaultScenario)
ScenarioRegistry.register(IbrNightmareScenario)
ScenarioRegistry.register(IbrMultiEventScenario)
ScenarioRegistry.register(UnbalancedSagScenario)
ScenarioRegistry.register(OOBIInterferenceScenario)
ScenarioRegistry.register(G5_E19_Phase_Jump_Sweep)
ScenarioRegistry.register(G5_E20_SNR_Sweep)
