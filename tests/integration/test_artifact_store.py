"""Integration test: ArtifactStore save/load."""
from __future__ import annotations

from pathlib import Path
from openfreqbench.io.artifact_store import ArtifactStore, ArtifactIdentity


def test_artifact_store_save_and_exists(tmp_path):
    store = ArtifactStore(root=tmp_path)
    identity = ArtifactIdentity(
        scenario_id="G1_E1",
        method_id="ZC",
        seed=0,
        scenario_params_hash="aabbccdd",
        method_params_hash="11223344",
    )
    data = {"RMSE_HZ": 0.01}
    store.save_json(identity, "report", data)
    assert store.exists(identity, "report")
