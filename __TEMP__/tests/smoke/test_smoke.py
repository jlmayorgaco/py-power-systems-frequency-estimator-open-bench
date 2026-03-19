"""
Smoke test: end-to-end pipeline in < 30 seconds.

Exercises: scenario → estimator → metrics → aggregation.
Does NOT require full Monte Carlo (uses 3 seeds only).
"""

import math

import numpy as np
import pytest

from openfreqbench.estimators.zc import ZeroCrossingEstimator
from openfreqbench.metrics.aggregate import aggregate_monte_carlo
from openfreqbench.metrics.frequency import MetricConfig, compute_metrics
from openfreqbench.profiling.timing import TimingHarness
from openfreqbench.runners.trace_runner import TraceRunner
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz


FS = 10_000.0
T_S = 0.5  # short waveform for speed
F_NOM = 60.0
N_SEEDS = 3


@pytest.fixture
def scenario():
    return G1_E1_Pure_60Hz(fs_hz=FS, T_s=T_S, f_nom_hz=F_NOM)


@pytest.fixture
def estimator():
    return ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})


@pytest.fixture
def metric_cfg():
    return MetricConfig(fs_hz=FS, f_nom=F_NOM, warm_up_s=0.05)


class TestSmokePipeline:
    def test_scenario_builds(self, scenario):
        out = scenario.build()
        assert out.v.shape[0] == int(T_S * FS)
        assert not np.any(np.isnan(out.v))

    def test_estimator_runs(self, scenario, estimator):
        out = scenario.build()
        f_hat = estimator.run(out.v)
        assert f_hat.shape == out.v.shape

    def test_timing_harness(self, scenario, estimator):
        out = scenario.build()
        harness = TimingHarness()
        f_hat, exec_s = harness.timed_run(estimator, out.v)
        assert exec_s > 0.0
        assert f_hat.shape == out.v.shape

    def test_compute_metrics_returns_dict(self, scenario, estimator, metric_cfg):
        out = scenario.build()
        harness = TimingHarness()
        f_hat, exec_s = harness.timed_run(estimator, out.v)
        metrics = compute_metrics(
            f_hat=f_hat,
            f_true=out.f_true,
            exec_time_s=exec_s,
            latency_samples=estimator.latency_samples,
            cfg=metric_cfg,
            scenario_id=out.scenario_id,
        )
        assert "RMSE_HZ" in metrics
        assert "FE_MAX_MHZ" in metrics
        assert "TIME_PER_SAMPLE_US" in metrics

    def test_rmse_is_finite_and_small(self, scenario, estimator, metric_cfg):
        out = scenario.build()
        harness = TimingHarness()
        f_hat, exec_s = harness.timed_run(estimator, out.v)
        metrics = compute_metrics(
            f_hat=f_hat,
            f_true=out.f_true,
            exec_time_s=exec_s,
            latency_samples=estimator.latency_samples,
            cfg=metric_cfg,
            scenario_id=out.scenario_id,
        )
        rmse_val = metrics["RMSE_HZ"]["value"]
        assert rmse_val is not None
        assert math.isfinite(float(rmse_val))
        assert float(rmse_val) < 0.1  # < 100 mHz for clean pure sine

    def test_trace_runner(self, scenario, metric_cfg):
        runner = TraceRunner(cfg=metric_cfg)
        estimator = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})
        result = runner.run(scenario, estimator, seed=0)
        assert result.scenario_id == "G1_E1_Pure_60Hz"
        assert result.method_id == "ZeroCrossing"
        assert result.f_hat.shape == result.f_true.shape
        assert math.isfinite(result.exec_time_s)

    def test_monte_carlo_aggregation(self, scenario, metric_cfg):
        runner = TraceRunner(cfg=metric_cfg)
        runs_metrics = []
        for seed in range(N_SEEDS):
            est = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})
            result = runner.run(scenario, est, seed=seed)
            runs_metrics.append(result.metrics)

        agg = aggregate_monte_carlo(runs_metrics, cfg=metric_cfg)
        assert "RMSE_HZ" in agg
        assert agg["RMSE_HZ"]["n"] == N_SEEDS
        assert math.isfinite(agg["RMSE_HZ"]["mean"])

    def test_registry_lookup(self):
        from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry

        assert "ZeroCrossing" in EstimatorRegistry.list_names()
        assert "G1_E1_Pure_60Hz" in ScenarioRegistry.list_names()
