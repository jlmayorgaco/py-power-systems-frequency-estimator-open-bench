"""
openfreqbench/cli/app.py

Main Typer application — entry point for the `ofb` CLI command.
"""

from __future__ import annotations

from rich.console import Console
import typer

from openfreqbench._version import __version__
from openfreqbench.cli.commands.analyze import analyze_cmd
from openfreqbench.cli.commands.doctor import doctor_cmd
from openfreqbench.cli.commands.list_items import list_cmd
from openfreqbench.cli.commands.run import run_cmd
from openfreqbench.cli.commands.scaffold import scaffold_cmd
from openfreqbench.cli.commands.smoke import smoke_cmd
from openfreqbench.cli.commands.status import status_cmd

app = typer.Typer(
    name="ofb",
    help="OpenFreqBench - frequency estimator benchmarking CLI.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

app.command("run")(run_cmd)
app.command("list")(list_cmd)
app.command("doctor")(doctor_cmd)
app.command("status")(status_cmd)
app.command("scaffold")(scaffold_cmd)
app.command("analyze")(analyze_cmd)
app.command("smoke")(smoke_cmd)


@app.command("version")
def version_cmd() -> None:
    """Print version and exit."""
    console.print(f"openfreqbench {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
