"""ScenarioMethodRunner: Monte Carlo loop for one scenario × estimator pair."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.trace_runner import TraceResult, TraceRunner
from openfreqbench.scenarios._base import ScenarioBase
from openfreqbench.stats.aggregate import aggregate_monte_carlo


@dataclass
class ScenarioMethodResult:
    scenario_id: str
    method_id: str
    n_runs: int
    traces: List[TraceResult]
    aggregated: Dict[str, Dict[str, float]]
    method_params: Dict[str, Any] = field(default_factory=dict)


class ScenarioMethodRunner:
    def __init__(self, n_runs: int = 100, seed_start: int = 0, cfg: Optional[MetricConfig] = None) -> None:
        self.n_runs = n_runs; self.seed_start = seed_start
        self._trace_runner = TraceRunner(cfg=cfg)

    def run(self, scenario: ScenarioBase, estimator: BaseEstimator) -> ScenarioMethodResult:
        traces = [self._trace_runner.run(scenario, estimator, seed=self.seed_start + i) for i in range(self.n_runs)]
        agg = aggregate_monte_carlo([t.metrics for t in traces], cfg=self._trace_runner.cfg)
        return ScenarioMethodResult(scenario_id=traces[0].scenario_id, method_id=traces[0].method_id,
                                    n_runs=len(traces), traces=traces, aggregated=agg,
                                    method_params=estimator._params.copy())
