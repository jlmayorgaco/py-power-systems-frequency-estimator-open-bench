"""
openfreqbench/runners/scenario_method_runner.py

ScenarioMethodRunner: Monte Carlo loop for one (scenario × estimator) pair.

Responsibilities
----------------
1. Run N seeds via TraceRunner
2. Aggregate into Monte Carlo stats
3. Return a ScenarioMethodResult

NOT responsible for: parallelism, persistence, or tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.metrics.aggregate import aggregate_monte_carlo
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.trace_runner import TraceResult, TraceRunner
from openfreqbench.scenarios._base import ScenarioBase


@dataclass
class ScenarioMethodResult:
    """Aggregated result for N seeds of one scenario × estimator pair."""

    scenario_id: str
    method_id: str
    n_runs: int
    traces: List[TraceResult]
    aggregated: Dict[str, Dict[str, float]]
    method_params: Dict[str, Any] = field(default_factory=dict)


class ScenarioMethodRunner:
    """
    Monte Carlo loop: run one (scenario × estimator) pair over N seeds.

    Usage
    -----
        runner = ScenarioMethodRunner(n_runs=20)
        result = runner.run(scenario, estimator, seed_start=0)
    """

    def __init__(
        self,
        n_runs: int = 100,
        seed_start: int = 0,
        cfg: Optional[MetricConfig] = None,
    ) -> None:
        self.n_runs = n_runs
        self.seed_start = seed_start
        self._trace_runner = TraceRunner(cfg=cfg)

    def run(
        self,
        scenario: ScenarioBase,
        estimator: BaseEstimator,
    ) -> ScenarioMethodResult:
        traces: List[TraceResult] = []

        for i in range(self.n_runs):
            seed = self.seed_start + i
            trace = self._trace_runner.run(scenario, estimator, seed=seed)
            traces.append(trace)

        metric_dicts = [t.metrics for t in traces]
        agg = aggregate_monte_carlo(metric_dicts, cfg=self._trace_runner.cfg)

        return ScenarioMethodResult(
            scenario_id=traces[0].scenario_id,
            method_id=traces[0].method_id,
            n_runs=len(traces),
            traces=traces,
            aggregated=agg,
            method_params=estimator._params.copy(),
        )
