"""
openfreqbench/runners/suite_runner.py

Parallel Execution Engine for large-scale performance benchmarking over tuning/testing suites.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any

from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry
from openfreqbench.runners.trace_runner import TraceRunner, TraceResult

@dataclass
class SuiteRunner:
    """
    Executes a predefined suite (e.g., all tuning scenarios x all estimators)
    using multi-processing for performance.
    """
    parallel: bool = True
    max_workers: int | None = None

    def run_tuning_suite(self) -> list[TraceResult]:
        scenarios = ScenarioRegistry.list_tuning_names()
        estimators = EstimatorRegistry.list_names()
        return self._execute_grid(scenarios, estimators)

    def run_testing_suite(self) -> list[TraceResult]:
        scenarios = ScenarioRegistry.list_testing_names()
        estimators = EstimatorRegistry.list_names()
        return self._execute_grid(scenarios, estimators)

    def _execute_grid(self, scenarios: list[str], estimators: list[str]) -> list[TraceResult]:
        tasks = []
        for s_id in scenarios:
            for e_id in estimators:
                tasks.append((s_id, e_id))

        if not self.parallel:
            return [self._run_single_task(t) for t in tasks]

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_single_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    # Logging could be injected here
                    pass
        return results

    @staticmethod
    def _run_single_task(task: tuple[str, str]) -> TraceResult | None:
        s_id, e_id = task
        try:
            scenario_cls = ScenarioRegistry.get(s_id)
            estimator_cls = EstimatorRegistry.get(e_id)
            runner = TraceRunner()
            return runner.run(scenario=scenario_cls(), estimator=estimator_cls(), seed=42)
        except NotImplementedError:
            # Skip scaffolding or unimplemented elements
            return None
        except Exception:
            # Other errors generally should be logged or bubbled, we suppress for parallel resiliency
            return None
