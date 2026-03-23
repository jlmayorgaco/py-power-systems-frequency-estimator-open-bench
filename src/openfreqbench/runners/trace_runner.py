"""
openfreqbench/runners/trace_runner.py

TraceRunner: atomic execution unit — one scenario x method x seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.metrics.frequency import MetricConfig, compute_metrics
from openfreqbench.profiling.timing import TimingHarness
from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class TraceResult:
    scenario_id: str
    method_id: str
    seed: int
    f_hat: np.ndarray
    f_true: np.ndarray
    exec_time_s: float
    latency_samples: int
    metrics: dict[str, Any]
    method_params: dict[str, Any] = field(default_factory=dict)


class TraceRunner:
    def __init__(self, cfg: MetricConfig | None = None) -> None:
        self.cfg = cfg
        self._harness = TimingHarness()

    def run(self, scenario: ScenarioBase, estimator: BaseEstimator, seed: int = 0) -> TraceResult:
        scenario = scenario.set_montecarlo_tuning({"seed": seed})
        waveform: ScenarioOutput = scenario.build()
        cfg = self.cfg or MetricConfig(
            fs_hz=float(waveform.state.fs_hz),
            f_nom=float(waveform.state.f_nom_hz),
        )
        if "fs" in estimator._params or hasattr(estimator, "_fs"):
            estimator.set_params(fs=float(waveform.state.fs_hz))
        f_hat, exec_time_s = self._harness.timed_run(estimator, waveform.v)
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
