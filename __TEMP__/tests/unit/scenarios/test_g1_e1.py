"""Unit tests for G1_E1_Pure_60Hz scenario."""

import numpy as np
import pytest

from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz
from openfreqbench.scenarios._base import ScenarioOutput


class TestG1E1Pure60Hz:
    def test_build_returns_scenario_output(self):
        sc = G1_E1_Pure_60Hz()
        out = sc.build()
        assert isinstance(out, ScenarioOutput)

    def test_scenario_id(self):
        sc = G1_E1_Pure_60Hz()
        out = sc.build()
        assert out.scenario_id == "G1_E1_Pure_60Hz"

    def test_array_lengths_consistent(self):
        sc = G1_E1_Pure_60Hz(fs_hz=1000.0, T_s=2.0)
        out = sc.build()
        n = 2000
        assert out.t.shape == (n,)
        assert out.v.shape == (n,)
        assert out.f_true.shape == (n,)

    def test_f_true_constant_at_nominal(self):
        sc = G1_E1_Pure_60Hz(f_nom_hz=60.0)
        out = sc.build()
        np.testing.assert_allclose(out.f_true, 60.0)

    def test_voltage_is_sine(self):
        sc = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=1.0, A0=2.0, f_nom_hz=60.0, phi0_rad=0.0)
        out = sc.build()
        expected = 2.0 * np.sin(2.0 * np.pi * 60.0 * out.t)
        np.testing.assert_allclose(out.v, expected, atol=1e-10)

    def test_sampling_rate(self):
        sc = G1_E1_Pure_60Hz(fs_hz=5_000.0, T_s=1.0)
        out = sc.build()
        dt = np.diff(out.t)
        np.testing.assert_allclose(dt, 1.0 / 5_000.0, rtol=1e-9)

    def test_seed_reproducibility(self):
        sc1 = G1_E1_Pure_60Hz(seed=42)
        sc2 = G1_E1_Pure_60Hz(seed=42)
        out1 = sc1.build()
        out2 = sc2.build()
        np.testing.assert_array_equal(out1.v, out2.v)

    def test_montecarlo_tuning_seed(self):
        sc = G1_E1_Pure_60Hz()
        sc.set_montecarlo_tuning({"seed": 99})
        assert sc.seed == 99

    def test_montecarlo_tuning_freq(self):
        sc = G1_E1_Pure_60Hz(f_nom_hz=60.0)
        sc.set_montecarlo_tuning({"f": 59.8})
        out = sc.build()
        np.testing.assert_allclose(out.f_true, 59.8)

    def test_invalid_alias_raises(self):
        sc = G1_E1_Pure_60Hz()
        with pytest.raises(ValueError, match="not in tuning_map"):
            sc.set_montecarlo_tuning({"invalid_alias": 1.0})

    def test_schema_fields(self):
        sc = G1_E1_Pure_60Hz(seed=7)
        out = sc.build()
        schema = out.state.schema
        assert schema["scenario_id"] == "G1_E1_Pure_60Hz"
        assert schema["seed"] == 7
        assert "fs_hz" in schema
