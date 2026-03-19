"""Config fixtures for tests."""
from __future__ import annotations

SMOKE_CONFIG_DICT = {
    "benchmark": {"name": "smoke", "output_dir": "artifacts", "n_runs": 3, "seed_start": 0},
    "scenarios": [{"id": "G1_E1_Pure_60Hz", "params": {"fs_hz": 10000, "T_s": 0.5}}],
    "estimators": [{"id": "ZeroCrossing", "params": {}}],
    "metrics": {"fs_hz": 10000, "f_nom": 60},
}
