from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ofb.cli.libs.errors import echo_err
from ofb.cli.libs.imports import lazy_bench


def register_bench_commands(app: typer.Typer) -> None:
    """Register benchmark-related commands on the given Typer app."""

    @app.command("plan")
    def plan_cmd(
        cfg: Path = typer.Argument(
            ...,
            exists=True,
            readable=True,
            help="Benchmark YAML configuration file",
        ),
        expand_only: bool = typer.Option(
            False,
            "--expand-only",
            help="Print expanded run matrix and exit (no execution)",
        ),
    ) -> None:
        """Expand a benchmark config into a concrete run matrix (no execution)."""
        load_cfg, plan_matrix, _run_matrix = lazy_bench()
        cfg_obj = load_cfg(cfg)
        matrix = plan_matrix(cfg_obj)
        if expand_only:
            typer.echo(json.dumps(matrix, indent=2))
            return
        typer.secho(f"Planned {len(matrix)} runs.", fg=typer.colors.GREEN)

    @app.command("dry-run")
    def dry_run_cmd(
        cfg: Path = typer.Argument(
            ...,
            exists=True,
            readable=True,
            help="Benchmark YAML configuration file",
        ),
    ) -> None:
        """Validate config, expand matrix, and print the plan; do not execute."""
        load_cfg, plan_matrix, _run_matrix = lazy_bench()
        cfg_obj = load_cfg(cfg)
        matrix = plan_matrix(cfg_obj)

        typer.secho("DRY RUN PLAN", fg=typer.colors.CYAN, bold=True)
        for i, row in enumerate(matrix, 1):
            typer.echo(
                f"[{i:03d}] scenario={row['scenario']} "
                f"estimator={row['estimator']} trials={row.get('trials', 1)}"
            )
        typer.secho(f"\nTotal planned: {len(matrix)} runs", fg=typer.colors.GREEN)

    @app.command("run-benchmark")
    def run_benchmark_cmd(
        cfg: Path = typer.Argument(
            ...,
            exists=True,
            readable=True,
            help="Benchmark YAML configuration file",
        ),
        limit: Optional[int] = typer.Option(
            None,
            "--limit",
            help="Limit the number of runs taken from the expanded matrix",
        ),
        outdir: Path = typer.Option(
            Path("results"),
            "--outdir",
            "-o",
            help="Output directory to store run artifacts",
        ),
        seed: Optional[int] = typer.Option(
            None,
            "--seed",
            help="Global random seed",
        ),
        fail_fast: bool = typer.Option(
            False,
            "--fail-fast",
            help="Stop on first error",
        ),
    ) -> None:
        """Execute the expanded run matrix and persist artifacts under results/."""
        load_cfg, plan_matrix, run_matrix = lazy_bench()
        cfg_obj = load_cfg(cfg)
        matrix = plan_matrix(cfg_obj)

        if limit is not None:
            matrix = matrix[:limit]

        outdir.mkdir(parents=True, exist_ok=True)
        ok = run_matrix(matrix, outdir=outdir, seed=seed, fail_fast=fail_fast)
        if not ok:
            echo_err("One or more runs failed.", code=2)

        typer.secho("Benchmark completed.", fg=typer.colors.GREEN)
