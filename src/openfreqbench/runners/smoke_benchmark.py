"""
openfreqbench/runners/smoke_benchmark.py

SmokeBenchmarkRunner — N×M benchmark grid with per-pair grid-search tuning
and Monte Carlo perturbations.

Pipeline per (scenario, estimator) pair:
  1. Grid-search best estimator params using n_tune_eval seeds.
  2. Apply best params to estimator; record TuningResult.
  3. Run n_runs MC seeds with optional scenario perturbations.
  4. Aggregate full statistical block (mean, std, CI, percentiles, CoV).

Design principles:
  - Framework orchestrates; estimator.tuning_ranges() declares search space.
  - Scenario perturbation policy is separate from estimator tuning space.
  - Perturbations are relative to a snapshot of nominal params (no drift).
  - All timing is external (TimingHarness via TraceRunner).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openfreqbench.core.config_models import MCPerturbSpec, ScenarioCfg, EstimatorCfg
from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.metrics.frequency import MetricConfig, rmse
from openfreqbench.runners.trace_runner import TraceResult, TraceRunner
from openfreqbench.scenarios._base import ScenarioBase
from openfreqbench.stats.aggregate import aggregate_monte_carlo


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """Records the outcome of grid-search tuning for one pair."""
    best_params: Dict[str, Any]
    best_score:  float            # RMSE_HZ on tuning seeds
    n_candidates: int             # total grid combos evaluated
    n_eval_seeds: int


@dataclass
class PairResult:
    """Results for one (scenario × estimator) pair."""
    scenario_id:    str
    method_id:      str
    n_runs:         int
    tuning:         TuningResult
    traces:         List[TraceResult]
    aggregated:     Dict[str, Dict[str, Any]]   # full stat blocks
    estimator_desc: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.tuning.best_params

    @property
    def tune_score(self) -> float:
        return self.tuning.best_score


@dataclass
class SmokeResult:
    """Full smoke benchmark result: all pairs + metadata."""
    name:       str
    pairs:      List[PairResult]
    scenarios:  List[str]
    estimators: List[str]

    def get_pair(self, scenario_id: str, method_id: str) -> Optional[PairResult]:
        for p in self.pairs:
            if p.scenario_id == scenario_id and p.method_id == method_id:
                return p
        return None

    def metric_grid(
        self,
        metric: str,
        stat: str = "mean",
    ) -> Dict[Tuple[str, str], Optional[float]]:
        """Return {(scenario_id, method_id): value} for a given metric/stat."""
        out: Dict[Tuple[str, str], Optional[float]] = {}
        for p in self.pairs:
            val = p.aggregated.get(metric, {}).get(stat)
            out[(p.scenario_id, p.method_id)] = val
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Grid-search helper
# ─────────────────────────────────────────────────────────────────────────────

def _grid_search(
    scenario:       ScenarioBase,
    estimator:      BaseEstimator,
    n_eval_seeds:   int,
    seed_start:     int,
    cfg:            MetricConfig,
    perturb_specs:  Dict[str, MCPerturbSpec],
) -> TuningResult:
    """
    Cartesian grid search over estimator.tuning_ranges().

    Evaluates each candidate config on n_eval_seeds MC waveforms (with
    perturbations), averages RMSE, returns TuningResult.

    The objective is estimator.suggested_objective() — defaults to RMSE_HZ.
    """
    definitions   = estimator.tuning_ranges()
    n_candidates  = 0

    if not definitions:
        return TuningResult(
            best_params=estimator._params.copy(),
            best_score=float("nan"),
            n_candidates=0,
            n_eval_seeds=n_eval_seeds,
        )

    param_names = [p.name for p in definitions]
    param_grids = [p.generate_grid() for p in definitions]
    combos      = list(itertools.product(*param_grids))
    n_candidates = len(combos)

    # Snapshot nominal params for perturbation reference
    nominal_vals: Dict[str, Any] = {}
    for alias, spec in perturb_specs.items():
        attr = scenario.tuning_map.get(alias, alias)
        if hasattr(scenario, attr):
            nominal_vals[attr] = getattr(scenario, attr)

    best_score = float("inf")
    best_cfg   = estimator._params.copy()

    for vals in combos:
        cfg_candidate = dict(zip(param_names, vals))
        estimator.set_params(**cfg_candidate)
        _inject_fs_hint(estimator, scenario)
        seed_scores = []
        for i in range(n_eval_seeds):
            seed = seed_start + i
            for attr, val in nominal_vals.items():
                setattr(scenario, attr, val)
            _apply_perturbations(scenario, perturb_specs, seed=seed,
                                 base_params=nominal_vals)
            try:
                waveform    = scenario.build()
                _inject_fs_hint(estimator, scenario)
                f_hat       = estimator.run(waveform.v)
                wu          = int(max(0.0, cfg.warm_up_s) * waveform.state.fs_hz)
                n_sig       = min(len(f_hat), len(waveform.f_true))
                if n_sig > wu + 2:
                    err   = f_hat[wu:n_sig] - waveform.f_true[wu:n_sig]
                    score = float(np.sqrt(np.mean(np.asarray(err, dtype=float) ** 2)))
                    if np.isfinite(score):
                        seed_scores.append(score)
            except Exception:
                continue

        mean_score = float(np.mean(seed_scores)) if seed_scores else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_cfg   = cfg_candidate.copy()

    # Restore nominal scenario state
    for attr, val in nominal_vals.items():
        setattr(scenario, attr, val)

    estimator.set_params(**best_cfg)
    return TuningResult(
        best_params=best_cfg,
        best_score=best_score,
        n_candidates=n_candidates,
        n_eval_seeds=n_eval_seeds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_perturbations(
    scenario:    ScenarioBase,
    specs:       Dict[str, MCPerturbSpec],
    seed:        int,
    base_params: Dict[str, Any],
) -> None:
    """
    Perturb scenario attributes in-place around nominal (base_params).

    Each spec: mode "rel" → delta ~ N(0, scale * |nominal|)
               mode "abs" → delta ~ N(0, scale)
    """
    if not specs:
        return
    rng = np.random.default_rng(seed & 0x7FFF_FFFF)
    for alias, spec in specs.items():
        attr    = scenario.tuning_map.get(alias, alias)
        if not hasattr(scenario, attr):
            continue
        nominal = float(base_params[attr]) if attr in base_params \
                  else float(getattr(scenario, attr))
        if spec.mode == "rel":
            delta = rng.normal(0.0, abs(spec.scale) * abs(nominal))
        else:
            delta = rng.normal(0.0, abs(spec.scale))
        setattr(scenario, attr, type(getattr(scenario, attr))(nominal + delta))


def _inject_fs_hint(estimator: BaseEstimator, scenario: ScenarioBase) -> None:
    """Inject fs_hz hint for FFT-based estimators that need it externally."""
    try:
        estimator._fs_hint = float(getattr(scenario, "fs_hz", 10_000.0))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

class SmokeBenchmarkRunner:
    """
    Runs all scenario × estimator pairs with grid-search + MC perturbations.

    Architecture:
        - Framework orchestrates the outer loop.
        - estimator.tuning_ranges() declares the search space.
        - scenario mc_perturbations declares per-seed variability.
        - TraceRunner handles one seed (timing + metrics).
        - aggregate_monte_carlo computes full statistical block.
    """

    def __init__(
        self,
        n_runs:      int = 30,
        seed_start:  int = 0,
        n_tune_eval: int = 5,
        cfg:         Optional[MetricConfig] = None,
        verbose:     bool = False,
    ) -> None:
        self.n_runs      = n_runs
        self.seed_start  = seed_start
        self.n_tune_eval = n_tune_eval
        self.cfg         = cfg or MetricConfig(fs_hz=10_000.0)
        self.verbose      = verbose
        self._trace_runner = TraceRunner(cfg=self.cfg)

    def run(
        self,
        scenario_cfgs:     List[ScenarioCfg],
        estimator_cfgs:    List[EstimatorCfg],
        scenario_registry,
        estimator_registry,
        name:              str  = "smoke",
        progress_cb        = None,
    ) -> SmokeResult:
        """
        Run the full N×M benchmark.

        progress_cb: optional callable(label: str) called before each pair.
        """
        pairs:         List[PairResult] = []
        scenario_ids   = [s.id for s in scenario_cfgs]
        estimator_ids  = [e.id for e in estimator_cfgs]

        for scen_cfg in scenario_cfgs:
            for est_cfg in estimator_cfgs:
                label = f"{scen_cfg.id} x {est_cfg.id}"
                if progress_cb:
                    progress_cb(label)

                try:
                    scenario  = scenario_registry.build(scen_cfg.id,
                                                        scen_cfg.params or None)
                    estimator = estimator_registry.build(est_cfg.id,
                                                         est_cfg.params or None)
                except KeyError as exc:
                    if self.verbose:
                        print(f"[WARN] Registry error for {label}: {exc}")
                    continue

                perturb_specs: Dict[str, MCPerturbSpec] = dict(
                    scen_cfg.mc_perturbations
                )
                _inject_fs_hint(estimator, scenario)

                # ── Step 1: Grid search ───────────────────────────────────────
                tuning = _grid_search(
                    scenario=scenario,
                    estimator=estimator,
                    n_eval_seeds=self.n_tune_eval,
                    seed_start=self.seed_start,
                    cfg=self.cfg,
                    perturb_specs=perturb_specs,
                )

                # ── Step 2: Reset scenario to nominal params ──────────────────
                if scen_cfg.params:
                    for k, v in scen_cfg.params.items():
                        if hasattr(scenario, k):
                            setattr(scenario, k,
                                    type(getattr(scenario, k))(v))

                # ── Step 3: MC runs with best params ──────────────────────────
                _inject_fs_hint(estimator, scenario)
                traces = self._run_mc(
                    scenario=scenario,
                    estimator=estimator,
                    perturb_specs=perturb_specs,
                    n_runs=self.n_runs,
                    seed_start=self.seed_start,
                )

                agg = aggregate_monte_carlo(
                    [t.metrics for t in traces],
                    cfg=self._trace_runner.cfg,
                )

                pairs.append(PairResult(
                    scenario_id=scen_cfg.id,
                    method_id=est_cfg.id,
                    n_runs=len(traces),
                    tuning=tuning,
                    traces=traces,
                    aggregated=agg,
                    estimator_desc=estimator.__class__.descriptor(),
                ))

        return SmokeResult(
            name=name,
            pairs=pairs,
            scenarios=scenario_ids,
            estimators=estimator_ids,
        )

    def _run_mc(
        self,
        scenario:      ScenarioBase,
        estimator:     BaseEstimator,
        perturb_specs: Dict[str, MCPerturbSpec],
        n_runs:        int,
        seed_start:    int,
    ) -> List[TraceResult]:
        """Run n_runs seeds with per-seed perturbations from nominal params."""
        # Snapshot nominal before loop
        nominal_vals: Dict[str, Any] = {}
        for alias in perturb_specs:
            attr = scenario.tuning_map.get(alias, alias)
            if hasattr(scenario, attr):
                nominal_vals[attr] = getattr(scenario, attr)

        traces = []
        for i in range(n_runs):
            seed = seed_start + i
            # Reset to nominal then perturb
            for attr, val in nominal_vals.items():
                setattr(scenario, attr, val)
            _apply_perturbations(scenario, perturb_specs, seed=seed,
                                 base_params=nominal_vals)
            _inject_fs_hint(estimator, scenario)
            tr = self._trace_runner.run(scenario, estimator, seed=seed)
            traces.append(tr)

        # Restore nominal
        for attr, val in nominal_vals.items():
            setattr(scenario, attr, val)

        return traces
