"""Integration test: ScenarioMethodRunner MC loop."""

from __future__ import annotations

from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.scenario_method_runner import ScenarioMethodRunner
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz


def test_scenario_method_runner_three_seeds():
    scenario = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)
    estimator = ZeroCrossingEstimator()
    cfg = MetricConfig(fs_hz=10_000.0, f_nom=60.0)
    runner = ScenarioMethodRunner(n_runs=3, seed_start=0, cfg=cfg)
    result = runner.run(scenario, estimator)
    assert result.n_runs == 3
