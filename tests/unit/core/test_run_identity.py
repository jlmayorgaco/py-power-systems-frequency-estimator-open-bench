"""Unit tests for RunIdentity / ArtifactIdentity."""
from __future__ import annotations

from openfreqbench.core.run_identity import RunIdentity


def test_cache_key_is_deterministic():
    ri = RunIdentity.from_dicts(
        scenario_id="G1_E1",
        method_id="ZC",
        seed=0,
        scenario_params={"fs_hz": 10000},
        method_params={"filter_win": 5},
    )
    assert ri.cache_key == RunIdentity.from_dicts(
        scenario_id="G1_E1",
        method_id="ZC",
        seed=0,
        scenario_params={"fs_hz": 10000},
        method_params={"filter_win": 5},
    ).cache_key


def test_cache_key_changes_with_seed():
    ri0 = RunIdentity.from_dicts("G1_E1", "ZC", 0, {}, {})
    ri1 = RunIdentity.from_dicts("G1_E1", "ZC", 1, {}, {})
    assert ri0.cache_key != ri1.cache_key
