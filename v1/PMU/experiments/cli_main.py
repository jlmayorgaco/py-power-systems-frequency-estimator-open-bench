# experiments/cli_main.py
from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.build_registry import build_default_registry
from experiments.runner import BenchmarkRunner
from experiments.mc import MonteCarloRunner


def run_all(cfg: ExperimentConfig, signals):
    registry = build_default_registry()

    det = BenchmarkRunner(cfg, registry).run(signals)

    mc_out = None
    if cfg.mc is not None:
        mc_out = MonteCarloRunner(cfg, registry).run(signals)

    return det, mc_out
