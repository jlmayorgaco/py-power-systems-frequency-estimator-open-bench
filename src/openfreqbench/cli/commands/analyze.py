"""ofb analyze -- post-run analysis and comparison of existing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.table import Table
import typer

console = Console()


def analyze_cmd(
    artifacts_dir: Annotated[
        Path,
        typer.Argument(help="Artifacts directory to analyze."),
    ] = Path("artifacts"),
    metric: Annotated[
        str,
        typer.Option("--metric", "-m", help="Primary metric to rank by."),
    ] = "RMSE_HZ",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write Markdown report to this file."),
    ] = None,
) -> None:
    """Analyze computed artifacts: load all report.json files and print a ranking table."""
    root = Path(artifacts_dir)
    if not root.exists():
        console.print(f"[red]No artifacts directory:[/red] {root.resolve()}")
        raise typer.Exit(code=1)

    reports = sorted(root.rglob("*_report.json"))
    if not reports:
        console.print(f"[yellow]No report.json files found in {root.resolve()}[/yellow]")
        raise typer.Exit(code=0)

    rows = []
    for rpath in reports:
        try:
            data = json.loads(rpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Each report is an aggregated dict: {metric_name: {mean, std, ...}}
        agg = data if isinstance(data, dict) else {}
        # Infer scenario/method from directory structure
        parts = rpath.parts
        scenario = parts[-3] if len(parts) >= 3 else "?"
        method = parts[-2] if len(parts) >= 2 else "?"
        val = agg.get(metric, {}).get("mean", float("nan"))
        std = agg.get(metric, {}).get("std", float("nan"))
        tps = agg.get("TIME_PER_SAMPLE_US", {}).get("mean", float("nan"))
        n = agg.get(metric, {}).get("n", 0)
        rows.append((scenario, method, val, std, tps, n))

    # Sort by metric value ascending (lower = better for error metrics)
    rows.sort(key=lambda r: (r[2] != r[2], r[2]))  # NaN last

    table = Table(
        "Rank",
        "Scenario",
        "Estimator",
        f"{metric} mean",
        f"{metric} std",
        "us/sample",
        "n",
        title=f"Ranking by {metric}",
    )
    for i, (sc, m, v, s, tps, n) in enumerate(rows, 1):
        table.add_row(
            str(i),
            sc,
            m,
            f"{v:.6f}" if v == v else "nan",
            f"{s:.6f}" if s == s else "nan",
            f"{tps:.2f}" if tps == tps else "nan",
            str(n),
        )

    console.print(table)

    if output:
        lines = [
            f"# Analysis -- {metric}\n",
            f"Artifacts: `{root.resolve()}`\n\n",
            f"| Rank | Scenario | Estimator | {metric} mean | {metric} std | us/sample | n |\n",
            "|------|----------|-----------|--------------|--------------|-----------|---|\n",
        ]
        for i, (sc, m, v, s, tps, n) in enumerate(rows, 1):
            vf = f"{v:.6f}" if v == v else "nan"
            sf = f"{s:.6f}" if s == s else "nan"
            tf = f"{tps:.2f}" if tps == tps else "nan"
            lines.append(f"| {i} | {sc} | {m} | {vf} | {sf} | {tf} | {n} |\n")
        output.write_text("".join(lines), encoding="utf-8")
        console.print(f"[green]Report written:[/green] {output.resolve()}")
