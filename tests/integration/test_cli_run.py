"""Integration test: ofb run with smoke config."""
from __future__ import annotations

from typer.testing import CliRunner
from openfreqbench.cli.app import app


def test_cli_run_quick_smoke(tmp_path):
    import yaml
    config = {
        "benchmark": {"name": "test", "output_dir": str(tmp_path / "artifacts"), "n_runs": 2, "seed_start": 0},
        "scenarios": [{"id": "G1_E1_Pure_60Hz", "params": {"fs_hz": 10000, "T_s": 0.3}}],
        "estimators": [{"id": "ZeroCrossing", "params": {}}],
        "metrics": {"fs_hz": 10000, "f_nom": 60},
    }
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.dump(config))
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg_path)])
    assert result.exit_code == 0
