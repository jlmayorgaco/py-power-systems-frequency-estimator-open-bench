"""ofb run <config.yaml> — execute a benchmark suite."""

from __future__ import annotations

import contextlib
from pathlib import Path
import time
from typing import Annotated

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import typer

from openfreqbench.core.checkpoint import CheckpointManager
from openfreqbench.core.config_models import load_config
from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry
from openfreqbench.io.artifact_store import ArtifactIdentity, ArtifactStore
from openfreqbench.metrics.frequency import MetricConfig
from openfreqbench.runners.scenario_method_runner import ScenarioMethodResult, ScenarioMethodRunner

console = Console()


def run_cmd(
    config: Annotated[Path, typer.Argument(help="Path to benchmark.yaml")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config and exit.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Override output directory."),
    ] = None,
    plots: Annotated[
        bool,
        typer.Option(
            "--plots",
            help="Generate waveform, tracking, and error plots.",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Auto-resume from checkpoint (skip completed pairs).",
        ),
    ] = False,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Ignore checkpoint and restart from scratch.",
        ),
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show checkpoint status and exit."),
    ] = False,
) -> None:
    """Run a benchmark suite defined in a YAML configuration file."""
    if not config.exists():
        console.print(f"[red]Config not found:[/red] {config}")
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[red]Config parse error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    out_dir = output or Path(cfg.benchmark.output_dir)
    store = ArtifactStore(out_dir)
    do_plots = plots or cfg.benchmark.plots

    # ── Checkpoint setup ──────────────────────────────────────────────────────
    checkpoint = CheckpointManager(out_dir)
    n_total = len(cfg.scenarios) * len(cfg.estimators)

    if status:
        _print_checkpoint_status(checkpoint, n_total)
        raise typer.Exit(code=0)

    run_mode = checkpoint.startup_dialog(
        config=cfg,
        n_total=n_total,
        auto_resume=resume,
        auto_restart=restart,
    )

    console.print(f"\n[bold cyan]OpenFreqBench[/bold cyan] — {cfg.benchmark.name}")
    console.print(f"  Scenarios : {[s.id for s in cfg.scenarios]}")
    console.print(f"  Estimators: {[e.id for e in cfg.estimators]}")
    console.print(f"  n_runs    : {cfg.benchmark.n_runs}")
    console.print(f"  plots     : {do_plots}")
    console.print(f"  output    : {out_dir}")
    console.print(f"  mode      : {run_mode}\n")

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

    results_table = Table(
        "Scenario",
        "Estimator",
        "RMSE_HZ (mean)",
        "FE_MAX_MHZ (mean)",
        "Time/sample (us)",
    )
    all_ok = True

    for scen_cfg in cfg.scenarios:
        for est_cfg in cfg.estimators:
            label = f"{scen_cfg.id} x {est_cfg.id}"

            # ── Checkpoint: skip already-completed pairs ───────────────────
            if not checkpoint.should_run(est_cfg.id, scen_cfg.id, run_mode):
                console.print(f"  [dim]Skipping (checkpoint)[/dim] {label}")
                continue

            checkpoint.mark_started(est_cfg.id, scen_cfg.id)

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
                    checkpoint.mark_failed(est_cfg.id, scen_cfg.id, str(exc))
                    all_ok = False
                    continue

                try:
                    runner = ScenarioMethodRunner(
                        n_runs=cfg.benchmark.n_runs,
                        seed_start=cfg.benchmark.seed_start,
                        cfg=metric_cfg,
                    )
                    result = runner.run(scenario, estimator)
                except Exception as exc:
                    console.print(f"[red]Run error:[/red] {label}: {exc}")
                    checkpoint.mark_failed(est_cfg.id, scen_cfg.id, str(exc))
                    all_ok = False
                    continue

            ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

            identity = ArtifactIdentity.from_dicts(
                scenario_id=result.scenario_id,
                method_id=result.method_id,
                seed=cfg.benchmark.seed_start,
                scenario_params=scen_cfg.params or {},
                method_params=est_cfg.params or {},
            )

            # ── Build enhanced JSON report ─────────────────────────────────
            report_data = _build_report(result, cfg.benchmark.n_runs, cfg.benchmark.seed_start)

            plot_paths: list[Path] = []
            if do_plots:
                plot_paths = _generate_plots(
                    scenario=scenario,
                    estimator=estimator,
                    result=result,
                    out_dir=out_dir / result.scenario_id / result.method_id / "plots",
                    ts=ts,
                    metric_cfg=metric_cfg,
                )
                report_data["plots"] = [str(p) for p in plot_paths]
                if plot_paths:
                    console.print(
                        f"  [green]Plots saved:[/green] {len(plot_paths)} files in "
                        f"{(out_dir / result.scenario_id / result.method_id / 'plots').resolve()}",
                    )

            saved = store.save_json(identity, "report", report_data, ts=ts)
            result_file = str(saved) if saved else ""

            # ── Checkpoint: mark completed atomically ─────────────────────
            checkpoint.mark_completed(
                est_cfg.id,
                scen_cfg.id,
                result_file=result_file,
                n_mc=cfg.benchmark.n_runs,
            )

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_checkpoint_status(checkpoint: CheckpointManager, n_total: int) -> None:
    """Print human-readable checkpoint status to the console."""
    state = checkpoint.load()
    if state is None:
        console.print("[yellow]No checkpoint found.[/yellow]")
        return
    n_done = checkpoint.count_completed()
    n_failed = checkpoint.count_failed()
    console.print("\n[bold]Checkpoint status[/bold]")
    console.print(f"  Session  : {checkpoint.session_id[:8]}...")
    console.print(f"  Started  : {checkpoint.started_at}")
    console.print(f"  Elapsed  : {checkpoint.elapsed}")
    console.print(f"  Progress : {n_done}/{n_total} pairs completed")
    if n_failed:
        console.print(f"  [red]Failed   : {n_failed} pair(s)[/red]")
        for pair, err in checkpoint.list_failed_pairs():
            console.print(f"    • {pair}: {err[:80]}")
    if checkpoint.in_progress:
        console.print(f"  [yellow]In-progress: {checkpoint.in_progress.get('pair')}[/yellow]")
    console.print()


def _build_report(result: ScenarioMethodResult, n_runs: int, seed_start: int) -> dict:
    """Build a rich JSON report including aggregated stats + per-seed key metrics."""
    per_seed = []
    for tr in result.traces:
        row = {"seed": tr.seed}
        for key in ("RMSE_HZ", "MAE_HZ", "BIAS_HZ", "FE_MAX_MHZ", "TIME_PER_SAMPLE_US"):
            v = tr.metrics.get(key, {})
            row[key] = v.get("value") if isinstance(v, dict) else None
        per_seed.append(row)

    return {
        "scenario_id": result.scenario_id,
        "method_id": result.method_id,
        "n_runs": n_runs,
        "seed_start": seed_start,
        "method_params": result.method_params,
        "aggregated": result.aggregated,
        "per_seed": per_seed,
    }


def _generate_plots(
    *,
    scenario,
    estimator,
    result: ScenarioMethodResult,
    out_dir: Path,
    ts: str,
    metric_cfg: MetricConfig,
) -> list[Path]:
    """Run a reference trace (seed=0), collect MC f_hats, generate 3 plots."""
    try:
        from openfreqbench.plotting.scenario_plots import generate_scenario_method_plots
        from openfreqbench.profiling.timing import TimingHarness
    except ImportError as exc:
        console.print(f"[yellow]Plot import error:[/yellow] {exc}")
        return []

    # Reference waveform: re-build scenario at seed=0
    with contextlib.suppress(Exception):
        scenario.set_montecarlo_tuning({"seed": 0})
    waveform = scenario.build()

    # Reference f_hat
    harness = TimingHarness()
    f_hat_ref, _ = harness.timed_run(estimator, waveform.v)
    latency = estimator.latency_samples

    # Collect all MC f_hats from traces
    f_hat_all = [tr.f_hat for tr in result.traces]

    try:
        paths = generate_scenario_method_plots(
            t=waveform.t,
            v=waveform.v,
            f_true=waveform.f_true,
            f_hat_ref=f_hat_ref,
            f_hat_all=f_hat_all,
            latency=latency,
            out_dir=out_dir,
            ts=ts,
            scenario_id=result.scenario_id,
            method_id=result.method_id,
            f_nom=float(metric_cfg.f_nom),
        )
        return paths
    except Exception as exc:
        console.print(f"[yellow]Plot generation error:[/yellow] {exc}")
        return []
