"""
ofb smoke <config.yaml>

Full smoke benchmark:
  - grid-search tuning per (scenario x estimator) pair
  - 30 Monte Carlo runs per pair with scenario perturbations
  - full statistical aggregation (mean/std/CI/percentiles/CoV)
  - IEEE-grade output figures
  - structured artifact layout: raw/ reports/ figures/

Output directory layout:
  <output_dir>/<benchmark_name>/<timestamp>/
    raw/
      <scenario_id>/<estimator_id>/
        seed_NNNN.json           ← per-seed raw trace metrics
    reports/
      <scenario_id>/<estimator_id>/
        report.json              ← full aggregated stats + tuning info
      summary.json               ← cross-pair comparison table
    figures/
      scenario_overview.png
      benchmark_grid.png
      mc_distributions.png
      metric_heatmap_rmse.png
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import time
from typing import Annotated, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import typer

from openfreqbench.core.config_models import load_config
from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.smoke_benchmark import (
    PairResult,
    SmokeBenchmarkRunner,
    SmokeResult,
)

console = Console()

_KEY_METRICS = (
    "RMSE_HZ",
    "MAE_HZ",
    "BIAS_HZ",
    "FE_MAX_MHZ",
    "TIME_PER_SAMPLE_US",
    "LATENCY_SAMPLES",
)


def smoke_cmd(
    config: Annotated[Path, typer.Argument(help="Path to benchmark YAML.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Override output root."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config and exit.")] = False,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip figure generation.")] = False,
) -> None:
    """Run a smoke benchmark: grid-search tuning + Monte Carlo per pair."""
    if not config.exists():
        console.print(f"[red]Config not found:[/red] {config}")
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[red]Config parse error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = Path(output or cfg.benchmark.output_dir)
    run_dir = root / cfg.benchmark.name / ts
    raw_dir = run_dir / "raw"
    rep_dir = run_dir / "reports"
    fig_dir = run_dir / "figures"
    for d in (raw_dir, rep_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]OpenFreqBench Smoke[/bold cyan] — {cfg.benchmark.name}")
    console.print(f"  Scenarios  : {[s.id for s in cfg.scenarios]}")
    console.print(f"  Estimators : {[e.id for e in cfg.estimators]}")
    console.print(f"  n_runs     : {cfg.benchmark.n_runs}")
    console.print(f"  n_tune_eval: {cfg.benchmark.n_tune_eval}")
    console.print(f"  output     : {run_dir}\n")

    if dry_run:
        console.print("[yellow]Dry-run — exiting.[/yellow]")
        return

    metric_cfg = MetricConfig(
        fs_hz=cfg.metrics.fs_hz,
        f_nom=cfg.metrics.f_nom,
        warm_up_s=cfg.metrics.warm_up_s,
        ieee_fe_limit_mhz=cfg.metrics.ieee_fe_limit_mhz,
        ieee_rfe_limit_hzs=cfg.metrics.ieee_rfe_limit_hzs,
    )

    runner = SmokeBenchmarkRunner(
        n_runs=cfg.benchmark.n_runs,
        seed_start=cfg.benchmark.seed_start,
        n_tune_eval=cfg.benchmark.n_tune_eval,
        cfg=metric_cfg,
        verbose=False,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task_id = prog.add_task("Starting...", total=None)

        def _on_pair(label: str) -> None:
            prog.update(task_id, description=label)

        result: SmokeResult = runner.run(
            scenario_cfgs=cfg.scenarios,
            estimator_cfgs=cfg.estimators,
            scenario_registry=ScenarioRegistry,
            estimator_registry=EstimatorRegistry,
            name=cfg.benchmark.name,
            progress_cb=_on_pair,
        )

    # ── Save raw per-seed traces ──────────────────────────────────────────────
    for pair in result.pairs:
        pair_raw = raw_dir / pair.scenario_id / pair.method_id
        pair_raw.mkdir(parents=True, exist_ok=True)
        for tr in pair.traces:
            seed_doc = {
                "scenario_id": tr.scenario_id,
                "method_id": tr.method_id,
                "seed": tr.seed,
                "exec_time_s": float(tr.exec_time_s),
                "latency_samples": int(tr.latency_samples),
                "metrics": {
                    k: v for k, v in tr.metrics.items() if isinstance(v, dict) and "value" in v
                },
            }
            fname = pair_raw / f"seed_{tr.seed:04d}.json"
            fname.write_text(
                json.dumps(seed_doc, indent=2, default=str),
                encoding="utf-8",
            )

    # ── Save per-pair aggregated reports ──────────────────────────────────────
    for pair in result.pairs:
        pair_rep = rep_dir / pair.scenario_id / pair.method_id
        pair_rep.mkdir(parents=True, exist_ok=True)

        report = _build_pair_report(pair, cfg.benchmark.n_runs, cfg.benchmark.seed_start, ts)
        (pair_rep / "report.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Save cross-pair summary ────────────────────────────────────────────────
    summary = _build_summary(result, cfg, ts)
    (rep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    # ── Rich summary table ─────────────────────────────────────────────────────
    _print_table(result)

    # ── Figures ───────────────────────────────────────────────────────────────
    do_plots = not no_plots and (cfg.benchmark.plots or True)
    if do_plots:
        _generate_smoke_plots(cfg, result, metric_cfg, fig_dir, ts)

    console.print(
        f"\n[bold green]Smoke complete.[/bold green]  Output: {run_dir.resolve()}",
    )
    console.print(f"  raw/     : {(raw_dir.stat().st_size if False else 'per-seed JSONs')}")
    console.print(f"  reports/ : {len(result.pairs)} pair reports + summary.json")
    console.print(f"  figures/ : {'saved' if do_plots else 'skipped'}")


# ─────────────────────────────────────────────────────────────────────────────
# Report builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_pair_report(
    pair: PairResult,
    n_runs: int,
    seed_start: int,
    ts: str,
) -> dict[str, Any]:
    per_seed = []
    for tr in pair.traces:
        row: dict[str, Any] = {"seed": tr.seed}
        for key in _KEY_METRICS:
            v = tr.metrics.get(key, {})
            row[key] = v.get("value") if isinstance(v, dict) else None
        per_seed.append(row)

    return {
        "schema_version": "2.0",
        "timestamp": ts,
        "scenario_id": pair.scenario_id,
        "method_id": pair.method_id,
        "n_runs": n_runs,
        "seed_start": seed_start,
        "estimator": pair.estimator_desc,
        "tuning": {
            "best_params": pair.tuning.best_params,
            "best_score_hz": pair.tuning.best_score,
            "n_candidates": pair.tuning.n_candidates,
            "n_eval_seeds": pair.tuning.n_eval_seeds,
        },
        "aggregated": pair.aggregated,
        "per_seed": per_seed,
    }


def _build_summary(result: SmokeResult, cfg, ts: str) -> dict[str, Any]:
    pairs_summary = []
    for pair in result.pairs:
        agg = pair.aggregated
        entry: dict[str, Any] = {
            "scenario_id": pair.scenario_id,
            "method_id": pair.method_id,
            "best_params": pair.best_params,
            "tune_score_hz": pair.tune_score,
            "n_runs": pair.n_runs,
        }
        for metric in ("RMSE_HZ", "MAE_HZ", "FE_MAX_MHZ", "TIME_PER_SAMPLE_US"):
            block = agg.get(metric, {})
            entry[f"{metric}_mean"] = block.get("mean")
            entry[f"{metric}_std"] = block.get("std")
            entry[f"{metric}_p95"] = block.get("p95")
            entry[f"{metric}_ci95_lo"] = block.get("ci95_lo")
            entry[f"{metric}_ci95_hi"] = block.get("ci95_hi")
            entry[f"{metric}_cov"] = block.get("cov")
        pairs_summary.append(entry)

    return {
        "schema_version": "2.0",
        "name": result.name,
        "timestamp": ts,
        "scenarios": result.scenarios,
        "estimators": result.estimators,
        "n_runs": cfg.benchmark.n_runs,
        "n_tune_eval": cfg.benchmark.n_tune_eval,
        "pairs": pairs_summary,
    }


def _print_table(result: SmokeResult) -> None:
    tbl = Table(
        "Scenario",
        "Estimator",
        "RMSE [mHz]\nmean±std",
        "p95 err [mHz]",
        "CI95 [mHz]",
        "CoV",
        "Time/samp [us]",
    )
    for pair in result.pairs:
        agg = pair.aggregated
        rmse_blk = agg.get("RMSE_HZ", {})
        tps_blk = agg.get("TIME_PER_SAMPLE_US", {})

        def _f(v, fmt=".3f"):
            return f"{v:{fmt}}" if (v is not None and v == v) else "nan"

        def _blk(d, k):
            v = d.get(k)
            return float(v) if v is not None else float("nan")

        mean_mhz = _blk(rmse_blk, "mean") * 1e3
        std_mhz = _blk(rmse_blk, "std") * 1e3
        p95_mhz = _blk(rmse_blk, "p95") * 1e3
        lo_mhz = _blk(rmse_blk, "ci95_lo") * 1e3
        hi_mhz = _blk(rmse_blk, "ci95_hi") * 1e3
        cov = _blk(rmse_blk, "cov")
        tps = _blk(tps_blk, "mean")

        tbl.add_row(
            pair.scenario_id,
            pair.method_id,
            f"{_f(mean_mhz)}±{_f(std_mhz)}",
            _f(p95_mhz),
            f"[{_f(lo_mhz)}, {_f(hi_mhz)}]",
            _f(cov, ".4f"),
            _f(tps),
        )
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_smoke_plots(
    cfg,
    result: SmokeResult,
    metric_cfg: MetricConfig,
    fig_dir: Path,
    ts: str,
) -> None:
    try:
        from openfreqbench.plotting.mc_summary import (
            plot_mc_distributions,
            plot_metric_heatmap,
        )
        from openfreqbench.plotting.suite_plots import (
            plot_benchmark_grid,
            plot_scenario_overview,
        )
        from openfreqbench.profiling.timing import TimingHarness
    except ImportError as exc:
        console.print(f"[yellow]Plot import error:[/yellow] {exc}")
        return

    harness = TimingHarness()
    scenario_data: list[dict] = []
    scenario_waveforms: dict[str, Any] = {}

    # Build reference waveforms (seed=0) for each scenario
    for scen_cfg in cfg.scenarios:
        try:
            scen = ScenarioRegistry.build(scen_cfg.id, scen_cfg.params or None)
            if "seed" in scen.tuning_map:
                scen = scen.set_montecarlo_tuning({"seed": 0})
            wf = scen.build()
            scenario_waveforms[scen_cfg.id] = wf
            scenario_data.append(
                {
                    "id": scen_cfg.id,
                    "t": wf.t,
                    "v": wf.v,
                    "f_true": wf.f_true,
                    "fs_hz": wf.state.fs_hz,
                    "f_nom_hz": wf.state.f_nom_hz,
                    "schema": wf.state.schema,
                },
            )
        except Exception as exc:
            console.print(f"[yellow]Scenario build error ({scen_cfg.id}):[/yellow] {exc}")

    # ── Scenario overview ──────────────────────────────────────────────────
    if scenario_data:
        try:
            p = plot_scenario_overview(
                scenario_data,
                fig_dir / "scenario_overview.png",
            )
            console.print(f"  [green]Figure:[/green] {p.name}")
        except Exception as exc:
            console.print(f"[yellow]Overview plot error:[/yellow] {exc}")

    # ── Benchmark grid ─────────────────────────────────────────────────────
    n_rows = len(result.scenarios)
    n_cols = len(result.estimators)
    grid_data: list[list] = [[None] * n_cols for _ in range(n_rows)]

    for row, scen_id in enumerate(result.scenarios):
        for col, est_id in enumerate(result.estimators):
            pair = result.get_pair(scen_id, est_id)
            wf = scenario_waveforms.get(scen_id)
            if pair is None or wf is None:
                continue
            try:
                est_obj = EstimatorRegistry.build(est_id, pair.best_params or None)
                with contextlib.suppress(Exception):
                    est_obj._fs_hint = wf.state.fs_hz
                f_hat_ref, _ = harness.timed_run(est_obj, wf.v)
                f_hat_all = [tr.f_hat for tr in pair.traces]
                agg = pair.aggregated

                grid_data[row][col] = {
                    "t": wf.t,
                    "f_true": wf.f_true,
                    "f_hat_ref": f_hat_ref,
                    "f_hat_all": f_hat_all,
                    "latency": est_obj.latency_samples,
                    "rmse_mean": agg.get("RMSE_HZ", {}).get("mean", float("nan")),
                    "rmse_std": agg.get("RMSE_HZ", {}).get("std", float("nan")),
                    "p95_err": agg.get("RMSE_HZ", {}).get("p95", float("nan")),
                    "f_nom": wf.state.f_nom_hz,
                    "schema": wf.state.schema,
                }
            except Exception as exc:
                console.print(f"[yellow]Grid cell ({scen_id}x{est_id}):[/yellow] {exc}")

    try:
        p = plot_benchmark_grid(
            grid_data,
            result.scenarios,
            result.estimators,
            fig_dir / "benchmark_grid.png",
        )
        console.print(f"  [green]Figure:[/green] {p.name}")
    except Exception as exc:
        console.print(f"[yellow]Grid plot error:[/yellow] {exc}")

    # ── MC distribution violin plot ────────────────────────────────────────
    try:
        p = plot_mc_distributions(result, fig_dir / "mc_distributions.png")
        console.print(f"  [green]Figure:[/green] {p.name}")
    except Exception as exc:
        console.print(f"[yellow]MC distribution plot error:[/yellow] {exc}")

    # ── RMSE heatmap ──────────────────────────────────────────────────────
    try:
        p = plot_metric_heatmap(
            result,
            fig_dir / "metric_heatmap_rmse.png",
            metric="RMSE_HZ",
            stat="mean",
        )
        console.print(f"  [green]Figure:[/green] {p.name}")
    except Exception as exc:
        console.print(f"[yellow]Heatmap plot error:[/yellow] {exc}")
