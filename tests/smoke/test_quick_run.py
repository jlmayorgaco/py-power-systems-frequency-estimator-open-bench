"""Smoke test: end-to-end TraceRunner pipeline."""

from __future__ import annotations

from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.trace_runner import TraceRunner
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz


def test_trace_runner_smoke():
    scenario = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)
    estimator = ZeroCrossingEstimator()
    cfg = MetricConfig(fs_hz=10_000.0, f_nom=60.0)
    runner = TraceRunner(cfg=cfg)
    result = runner.run(scenario, estimator, seed=0)
    assert result is not None
    assert "RMSE_HZ" in result.metrics
