"""ofb status — show current run state, artifact counts, and last run summary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.table import Table
import typer

console = Console()


def status_cmd(
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--dir",
            "-d",
            help="Artifacts directory to scan.",
        ),
    ] = Path("artifacts"),
) -> None:
    """Show what has been computed: scenarios x estimators x seed counts."""
    root = Path(artifacts_dir)
    if not root.exists():
        console.print(f"[yellow]No artifacts directory found at:[/yellow] {root.resolve()}")
        console.print("Run [bold]ofb run <config.yaml>[/bold] first.")
        raise typer.Exit(code=0)

    table = Table(
        "Scenario",
        "Estimator",
        "Reports",
        "Last run",
        title=f"Artifacts in {root.resolve()}",
    )
    found = False

    for scenario_dir in sorted(root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for method_dir in sorted(scenario_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            reports = sorted(method_dir.glob("*_report.json"))
            if not reports:
                continue
            found = True
            last_ts = reports[-1].stem.split("_report")[0].split("_")
            last_run = "_".join(last_ts[-2:]) if len(last_ts) >= 2 else "unknown"
            table.add_row(scenario_dir.name, method_dir.name, str(len(reports)), last_run)

    if found:
        console.print(table)
    else:
        console.print(f"[yellow]No report artifacts found in {root.resolve()}[/yellow]")
        console.print("Run [bold]ofb run <config.yaml>[/bold] to generate results.")
