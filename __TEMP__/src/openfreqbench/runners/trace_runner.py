"""
openfreqbench/runners/trace_runner.py

TraceRunner: the atomic execution unit — one scenario × one method × one seed.

Responsibilities
----------------
1. Build scenario waveform (build())
2. Run estimator with timing (TimingHarness.timed_run())
3. Compute full Q1 metrics (compute_metrics())
4. Return a TraceResult with all data attached

NOT responsible for: artifact persistence, MC aggregation, parallelism.
Those belong in ScenarioMethodRunner and above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.metrics.frequency import MetricConfig, compute_metrics
from openfreqbench.profiling.timing import TimingHarness
from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class TraceResult:
    """
    Full result from one (scenario × method × seed) run.

    Fields
    ------
    scenario_id   : str
    method_id     : str
    seed          : int
    f_hat         : estimated frequency trace [Hz]
    f_true        : ground-truth frequency trace [Hz]
    exec_time_s   : wall-clock time for estimator.run() [s]
    latency_samples: causal delay
    metrics       : full Q1 metric dict (from compute_metrics)
    method_params : params dict used for this run
    """

    scenario_id: str
    method_id: str
    seed: int
    f_hat: np.ndarray
    f_true: np.ndarray
    exec_time_s: float
    latency_samples: int
    metrics: Dict[str, Any]
    method_params: Dict[str, Any] = field(default_factory=dict)


class TraceRunner:
    """
    Runs one (scenario × estimator × seed) trial.

    Usage
    -----
        runner = TraceRunner(cfg=MetricConfig(fs_hz=10_000))
        result = runner.run(scenario, estimator, seed=0)
    """

    def __init__(self, cfg: Optional[MetricConfig] = None) -> None:
        self.cfg = cfg
        self._harness = TimingHarness()

    def run(
        self,
        scenario: ScenarioBase,
        estimator: BaseEstimator,
        seed: int = 0,
    ) -> TraceResult:
        """
        Execute one trial.

        Parameters
        ----------
        scenario  : ScenarioBase instance (will call set_montecarlo_tuning + build)
        estimator : BaseEstimator instance
        seed      : Monte Carlo seed

        Returns
        -------
        TraceResult
        """
        # 1. Inject seed into scenario and build waveform
        scenario.set_montecarlo_tuning({"seed": seed})
        waveform: ScenarioOutput = scenario.build()

        # 2. Determine MetricConfig (use provided or build from waveform)
        cfg = self.cfg or MetricConfig(
            fs_hz=float(waveform.state.fs_hz),
            f_nom=float(waveform.state.f_nom_hz),
        )

        # 3. Inject fs into estimator params if estimator uses it
        if "fs" in estimator._params or hasattr(estimator, "_fs"):
            estimator.set_params(fs=float(waveform.state.fs_hz))

        # 4. Timed run
        f_hat, exec_time_s = self._harness.timed_run(estimator, waveform.v)

        # 5. Compute metrics
        latency = estimator.latency_samples
        metrics = compute_metrics(
            f_hat=f_hat,
            f_true=waveform.f_true,
            exec_time_s=exec_time_s,
            latency_samples=latency,
            cfg=cfg,
            scenario_id=waveform.scenario_id,
        )

        return TraceResult(
            scenario_id=waveform.scenario_id,
            method_id=estimator.NAME,
            seed=seed,
            f_hat=f_hat,
            f_true=waveform.f_true,
            exec_time_s=exec_time_s,
            latency_samples=latency,
            metrics=metrics,
            method_params=estimator._params.copy(),
        )
