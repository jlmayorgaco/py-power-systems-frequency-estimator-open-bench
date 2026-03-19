from __future__ import annotations
import typer

def echo_err(msg: str, code: int = 1) -> None:
    """Consistent error printing + exit."""
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)
