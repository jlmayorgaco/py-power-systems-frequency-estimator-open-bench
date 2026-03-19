"""
openfreqbench/cli/commands/list_cmd.py

`ofb list` — show registered estimators and scenarios.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def list_cmd(
    kind: str = typer.Argument("all", help="'estimators', 'scenarios', or 'all'"),
) -> None:
    """List registered estimators and/or scenarios."""
    from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry  # lazy import

    show_est = kind in ("all", "estimators")
    show_scen = kind in ("all", "scenarios")

    if show_est:
        t = Table("Name", "Family", title="Registered Estimators")
        for name in EstimatorRegistry.list_names():
            cls = EstimatorRegistry.get(name)
            t.add_row(name, cls.FAMILY)
        console.print(t)

    if show_scen:
        t = Table("ID", title="Registered Scenarios")
        for name in ScenarioRegistry.list_names():
            t.add_row(name)
        console.print(t)

    if not show_est and not show_scen:
        console.print(f"[red]Unknown kind {kind!r}. Use 'estimators', 'scenarios', or 'all'.[/red]")
        raise typer.Exit(code=1)
