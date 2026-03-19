"""
openfreqbench/scenarios/registry.py

ScenarioRegistry: scenario_id → class mapping with built-in registrations.
"""
from __future__ import annotations
from typing import Any, Dict, Type
from openfreqbench.scenarios._base import ScenarioBase


class ScenarioRegistry:
    _registry: Dict[str, Type[ScenarioBase]] = {}

    @classmethod
    def register(cls, scenario_cls: Type[ScenarioBase]) -> Type[ScenarioBase]:
        cls._registry[scenario_cls.scenario_id] = scenario_cls
        return scenario_cls

    @classmethod
    def get(cls, name: str) -> Type[ScenarioBase]:
        if name not in cls._registry:
            raise KeyError(f"Scenario {name!r} not registered. Available: {sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def build(cls, name: str, params: Dict[str, Any] | None = None) -> ScenarioBase:
        cls_ = cls.get(name)
        return cls_(**params) if params else cls_()

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._registry)


from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz  # noqa: E402
from openfreqbench.scenarios.g2.e1_freq_step import G2_E1_FreqStep  # noqa: E402
from openfreqbench.scenarios.g2.e2_freq_ramp import G2_E2_FreqRamp  # noqa: E402

ScenarioRegistry.register(G1_E1_Pure_60Hz)
ScenarioRegistry.register(G2_E1_FreqStep)
ScenarioRegistry.register(G2_E2_FreqRamp)

# ── Scaffold scenario registrations (build() raises NotImplementedError) ──────
from openfreqbench.scenarios.g1.e2_gaussian_noise_1pct        import G1_E2_Gaussian_Noise_1pct        # noqa: E402
from openfreqbench.scenarios.g1.e3_gaussian_noise_5pct        import G1_E3_Gaussian_Noise_5pct        # noqa: E402
from openfreqbench.scenarios.g2.e4_voltage_mag_step_1pct      import G2_E4_Voltage_Mag_Step_1pct      # noqa: E402
from openfreqbench.scenarios.g2.e5_voltage_mag_step_10pct     import G2_E5_Voltage_Mag_Step_10pct     # noqa: E402
from openfreqbench.scenarios.g2.e6_freq_step_59p5             import G2_E6_Freq_Step_60_to_59p5       # noqa: E402
from openfreqbench.scenarios.g2.e7_freq_step_55               import G2_E7_Freq_Step_60_to_55         # noqa: E402
from openfreqbench.scenarios.g2.e8_fast_ramp                  import G2_E8_Fast_Ramp_plus5Hzs         # noqa: E402
from openfreqbench.scenarios.g2.e9_slow_ramp                  import G2_E9_Slow_Ramp_minus0p5Hzs      # noqa: E402
from openfreqbench.scenarios.g3.e10_am_modulation             import G3_E10_AM_Modulation             # noqa: E402
from openfreqbench.scenarios.g3.e11_fm_modulation             import G3_E11_FM_Modulation             # noqa: E402
from openfreqbench.scenarios.g3.e12_phase_jump                import G3_E12_Phase_Jump                # noqa: E402
from openfreqbench.scenarios.g3.e13_impulsive_outliers        import G3_E13_Impulsive_Outliers        # noqa: E402
from openfreqbench.scenarios.g3.e14_noise_harmonics           import G3_E14_Noise_Harmonics           # noqa: E402
from openfreqbench.scenarios.g3.e15_noise_interharmonics      import G3_E15_Noise_Interharmonics      # noqa: E402
from openfreqbench.scenarios.g4.e16_composite_islanding       import G4_E16_Composite_Islanding       # noqa: E402
from openfreqbench.scenarios.g4.e17_multi_event_profile       import G4_E17_Multi_Event_Profile       # noqa: E402
from openfreqbench.scenarios.g4.e18_chamorro_event            import G4_E18_Chamorro_Event            # noqa: E402
from openfreqbench.scenarios.g5.e19_phase_jump_sweep          import G5_E19_Phase_Jump_Sweep          # noqa: E402
from openfreqbench.scenarios.g5.e20_snr_sweep                 import G5_E20_SNR_Sweep                 # noqa: E402

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
ScenarioRegistry.register(G4_E18_Chamorro_Event)
ScenarioRegistry.register(G5_E19_Phase_Jump_Sweep)
ScenarioRegistry.register(G5_E20_SNR_Sweep)
