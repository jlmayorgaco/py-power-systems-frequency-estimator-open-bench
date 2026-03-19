"""
openfreqbench/cli/commands/run.py

`ofb run <config.yaml>` — execute a benchmark suite from a YAML config.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from openfreqbench.core.config import load_config
from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry
from openfreqbench.io.artifact_store import ArtifactIdentity, ArtifactStore
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.scenario_method_runner import ScenarioMethodRunner

console = Console()


def run_cmd(
    config: Path = typer.Argument(..., help="Path to benchmark.yaml"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and exit."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output directory."),
) -> None:
    """Run a benchmark suite defined in a YAML configuration file."""

    if not config.exists():
        console.print(f"[red]Config not found:[/red] {config}")
        raise typer.Exit(code=1)

    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[red]Config parse error:[/red] {exc}")
        raise typer.Exit(code=1)

    out_dir = output or Path(cfg.benchmark.output_dir)
    store = ArtifactStore(out_dir)

    console.print(f"\n[bold cyan]OpenFreqBench[/bold cyan] — {cfg.benchmark.name}")
    console.print(f"  Scenarios : {[s.id for s in cfg.scenarios]}")
    console.print(f"  Estimators: {[e.id for e in cfg.estimators]}")
    console.print(f"  n_runs    : {cfg.benchmark.n_runs}")
    console.print(f"  output    : {out_dir}\n")

    if dry_run:
        console.print("[yellow]Dry-run — exiting without execution.[/yellow]")
        return

    metric_cfg = MetricConfig(
        fs_hz=cfg.metrics.fs_hz,
        f_nom=cfg.metrics.f_nom,
        warm_up_s=cfg.metrics.warm_up_s,
        ieee_fe_limit_mhz=cfg.metrics.ieee_fe_limit_mhz,
        ieee_rfe_limit_hzs=cfg.metrics.ieee_rfe_limit_hzs,
    )

    results_table = Table("Scenario", "Estimator", "RMSE_HZ (mean)", "FE_MAX_MHZ (mean)", "Time/sample (µs)")
    all_ok = True

    for scen_cfg in cfg.scenarios:
        for est_cfg in cfg.estimators:
            label = f"{scen_cfg.id} × {est_cfg.id}"
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{label}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as prog:
                prog.add_task("running", total=None)

                try:
                    scenario = ScenarioRegistry.build(scen_cfg.id, scen_cfg.params or None)
                    estimator = EstimatorRegistry.build(est_cfg.id, est_cfg.params or None)
                except KeyError as exc:
                    console.print(f"[red]Registry error:[/red] {exc}")
                    all_ok = False
                    continue

                runner = ScenarioMethodRunner(
                    n_runs=cfg.benchmark.n_runs,
                    seed_start=cfg.benchmark.seed_start,
                    cfg=metric_cfg,
                )
                result = runner.run(scenario, estimator)

            # Persist
            identity = ArtifactIdentity.from_dicts(
                scenario_id=result.scenario_id,
                method_id=result.method_id,
                seed=cfg.benchmark.seed_start,
                scenario_params=scen_cfg.params or {},
                method_params=est_cfg.params or {},
            )
            ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            store.save_json(identity, "report", result.aggregated, ts=ts)

            # Summary row
            agg = result.aggregated
            rmse_mean = agg.get("RMSE_HZ", {}).get("mean", float("nan"))
            fe_mean = agg.get("FE_MAX_MHZ", {}).get("mean", float("nan"))
            tps_mean = agg.get("TIME_PER_SAMPLE_US", {}).get("mean", float("nan"))

            results_table.add_row(
                result.scenario_id,
                result.method_id,
                f"{rmse_mean:.6f}" if rmse_mean == rmse_mean else "nan",
                f"{fe_mean:.3f}" if fe_mean == fe_mean else "nan",
                f"{tps_mean:.2f}" if tps_mean == tps_mean else "nan",
            )

    console.print(results_table)

    if not all_ok:
        raise typer.Exit(code=1)
