"""OpenFreqBench CLI entrypoint.

Structure:
- Typer app with a root callback that can show a Rich banner.
- Separate command registrars under ofb.cli.commands.*
- Shared helpers under ofb.cli.libs.*
"""

from __future__ import annotations
from typing import Optional

import typer
from rich.console import Console
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn

from ofb.cli.libs.banner import build_banner
from ofb.cli.commands.registry_cmds import register_registry_commands
from ofb.cli.commands.bench_cmds import register_bench_commands
from ofb.cli.commands.report_cmds import register_report_commands

try:
    from ofb.version import __version__, PROJECT, AUTHOR
except Exception:
    __version__, PROJECT, AUTHOR = "0.0.0", "OpenFreqBench", "Unknown"


def create_app() -> tuple[typer.Typer, Console]:
    """Factory to build the Typer app and a Rich console (useful for testing)."""
    app = typer.Typer(add_completion=False, help=f"{PROJECT} CLI")
    console = Console()
    return app, console


app, console = create_app()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    banner: bool = typer.Option(True, "--banner/--no-banner", help="Show startup banner"),
    version: bool = typer.Option(False, "--version", "-V", help="Print version and exit"),
    max_width: Optional[int] = typer.Option(None, "--max-width", help="Cap banner width (cols)"),
    animate: bool = typer.Option(False, "--animate/--no-animate", help="Brief spinner before banner"),
):
    """Show banner + help when invoked without a subcommand."""
    if version:
        console.print(f"{PROJECT} {__version__}")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        if banner:
            if animate:
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[bold cyan]Preparing banner...[/]"),
                    transient=True,
                    console=console,
                ) as progress:
                    progress.add_task("prep", total=None)
            panel = build_banner(
                console,
                project=PROJECT,
                version=__version__,
                author=AUTHOR,
                max_width=max_width,
            )
            console.print(Align.center(panel, width=console.width))
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("banner")
def banner_cmd(
    max_width: Optional[int] = typer.Option(None, "--max-width", "-w", help="Cap banner width (cols)"),
    animate: bool = typer.Option(False, "--animate/--no-animate", help="Brief spinner before banner"),
):
    """Explicit command to render the banner (useful for demos/CI)."""
    if animate:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]Preparing banner...[/]"),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task("prep", total=None)
    panel = build_banner(
        console,
        project=PROJECT,
        version=__version__,
        author=AUTHOR,
        max_width=max_width,
    )
    console.print(Align.center(panel, width=console.width))


# Register feature command groups
register_registry_commands(app)
register_bench_commands(app)
register_report_commands(app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
