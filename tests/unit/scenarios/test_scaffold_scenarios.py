"""
tests/unit/scenarios/test_scaffold_scenarios.py

Parametrized smoke tests for all scaffold scenarios.

Each test verifies that:
  1. The class is importable and instantiable with no arguments.
  2. scenario_id is a non-empty string.
  3. build() raises NotImplementedError (scaffold contract).
  4. set_montecarlo_tuning() runs without error.
  5. The scenario is registered in ScenarioRegistry.

Scaffold scenarios are NOT expected to generate waveforms yet.
These tests only verify the *metadata contract* and *registration*.
"""
from __future__ import annotations

import pytest

# ── Scaffold scenario classes ──────────────────────────────────────────────────

from openfreqbench.scenarios.g1.e2_gaussian_noise_1pct        import G1_E2_Gaussian_Noise_1pct
from openfreqbench.scenarios.g1.e3_gaussian_noise_5pct        import G1_E3_Gaussian_Noise_5pct
from openfreqbench.scenarios.g2.e4_voltage_mag_step_1pct      import G2_E4_Voltage_Mag_Step_1pct
from openfreqbench.scenarios.g2.e5_voltage_mag_step_10pct     import G2_E5_Voltage_Mag_Step_10pct
from openfreqbench.scenarios.g2.e6_freq_step_59p5             import G2_E6_Freq_Step_60_to_59p5
from openfreqbench.scenarios.g2.e7_freq_step_55               import G2_E7_Freq_Step_60_to_55
from openfreqbench.scenarios.g2.e8_fast_ramp                  import G2_E8_Fast_Ramp_plus5Hzs
from openfreqbench.scenarios.g2.e9_slow_ramp                  import G2_E9_Slow_Ramp_minus0p5Hzs
from openfreqbench.scenarios.g3.e10_am_modulation             import G3_E10_AM_Modulation
from openfreqbench.scenarios.g3.e11_fm_modulation             import G3_E11_FM_Modulation
from openfreqbench.scenarios.g3.e12_phase_jump                import G3_E12_Phase_Jump
from openfreqbench.scenarios.g3.e13_impulsive_outliers        import G3_E13_Impulsive_Outliers
from openfreqbench.scenarios.g3.e14_noise_harmonics           import G3_E14_Noise_Harmonics
from openfreqbench.scenarios.g3.e15_noise_interharmonics      import G3_E15_Noise_Interharmonics
from openfreqbench.scenarios.g4.e16_composite_islanding       import G4_E16_Composite_Islanding
from openfreqbench.scenarios.g4.e17_multi_event_profile       import G4_E17_Multi_Event_Profile
from openfreqbench.scenarios.g4.e18_chamorro_event            import G4_E18_Chamorro_Event
from openfreqbench.scenarios.g5.e19_phase_jump_sweep          import G5_E19_Phase_Jump_Sweep
from openfreqbench.scenarios.g5.e20_snr_sweep                 import G5_E20_SNR_Sweep

SCAFFOLD_CLASSES = [
    G1_E2_Gaussian_Noise_1pct,
    G1_E3_Gaussian_Noise_5pct,
    G2_E4_Voltage_Mag_Step_1pct,
    G2_E5_Voltage_Mag_Step_10pct,
    G2_E6_Freq_Step_60_to_59p5,
    G2_E7_Freq_Step_60_to_55,
    G2_E8_Fast_Ramp_plus5Hzs,
    G2_E9_Slow_Ramp_minus0p5Hzs,
    G3_E10_AM_Modulation,
    G3_E11_FM_Modulation,
    G3_E12_Phase_Jump,
    G3_E13_Impulsive_Outliers,
    G3_E14_Noise_Harmonics,
    G3_E15_Noise_Interharmonics,
    G4_E16_Composite_Islanding,
    G4_E17_Multi_Event_Profile,
    G4_E18_Chamorro_Event,
    G5_E19_Phase_Jump_Sweep,
    G5_E20_SNR_Sweep,
]


def _scenario_id(cls):
    return cls().scenario_id


@pytest.mark.parametrize("cls", SCAFFOLD_CLASSES, ids=_scenario_id)
class TestScaffoldScenarioContract:
    """Metadata and stub-safety contract for all scaffold scenarios."""

    def test_instantiates_no_args(self, cls):
        s = cls()
        assert s is not None

    def test_scenario_id_nonempty(self, cls):
        s = cls()
        assert isinstance(s.scenario_id, str) and len(s.scenario_id) > 0

    def test_build_raises_not_implemented(self, cls):
        """Scaffold scenarios must not generate waveforms yet."""
        s = cls()
        with pytest.raises(NotImplementedError):
            s.build()

    def test_set_montecarlo_tuning_noop(self, cls):
        """set_montecarlo_tuning must accept a seed dict without crashing."""
        s = cls()
        result = s.set_montecarlo_tuning({"seed": 42})
        # returns self for chaining
        assert result is s

    def test_in_registry(self, cls):
        """Every scaffold scenario must be findable by scenario_id in ScenarioRegistry."""
        from openfreqbench.scenarios.registry import ScenarioRegistry
        s = cls()
        names = ScenarioRegistry.list_names()
        assert s.scenario_id in names, (
            f"{s.scenario_id} not found in ScenarioRegistry"
        )

    def test_fs_hz_positive(self, cls):
        s = cls()
        assert s.fs_hz > 0.0

    def test_T_s_positive(self, cls):
        s = cls()
        assert s.T_s > 0.0


# ── Registry coverage ──────────────────────────────────────────────────────────

def test_all_scaffold_scenarios_in_registry():
    from openfreqbench.scenarios.registry import ScenarioRegistry
    registered = set(ScenarioRegistry.list_names())
    expected = {cls().scenario_id for cls in SCAFFOLD_CLASSES}
    missing = expected - registered
    assert not missing, f"These scaffold scenarios are missing from registry: {missing}"


def test_total_registered_scenario_count():
    """Registry must have at least 22 scenarios (3 real + 19 scaffold)."""
    from openfreqbench.scenarios.registry import ScenarioRegistry
    count = len(ScenarioRegistry.list_names())
    assert count >= 22, f"Expected ≥22 registered scenarios, got {count}"
