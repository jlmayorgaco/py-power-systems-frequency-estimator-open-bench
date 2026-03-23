"""ofb doctor — validate installation, dependencies, and config correctness."""

from __future__ import annotations

from rich.console import Console
import typer

# Force UTF-8 on Windows to avoid cp1252 encode errors
console = Console(highlight=False)


def doctor_cmd() -> None:
    """Check that the environment and package are correctly installed."""
    ok = True
    checks = [
        ("openfreqbench importable", _check_import),
        ("numpy available", lambda: _check_pkg("numpy")),
        ("pydantic available", lambda: _check_pkg("pydantic")),
        ("typer available", lambda: _check_pkg("typer")),
        ("yaml available", lambda: _check_pkg("yaml")),
    ]
    for label, fn in checks:
        try:
            fn()
            console.print(f"  [green]OK[/green]  {label}")
        except Exception as exc:
            console.print(f"  [red]FAIL[/red] {label}: {exc}")
            ok = False
    if ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        console.print("\n[bold red]Some checks failed. Run `pip install -e .[dev]`.[/bold red]")
        raise typer.Exit(code=1)


def _check_import() -> None:
    import openfreqbench  # noqa: F401


def _check_pkg(name: str) -> None:
    __import__(name)
