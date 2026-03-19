"""Integration test: TraceRunner end-to-end."""
from __future__ import annotations

from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz
from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator
from openfreqbench.runners.trace_runner import TraceRunner
from openfreqbench.metrics.frequency import MetricConfig


def test_trace_runner_returns_result():
    scenario = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)
    estimator = ZeroCrossingEstimator()
    cfg = MetricConfig(fs_hz=10_000.0, f_nom=60.0)
    result = TraceRunner(cfg=cfg).run(scenario, estimator, seed=0)
    assert result.metrics.get("RMSE_HZ") is not None
