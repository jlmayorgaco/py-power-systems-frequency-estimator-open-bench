from __future__ import annotations
import json
import typer

from ..libs.errors import echo_err
from ..libs.imports import lazy_registry


SINGULAR_TO_PLURAL = {
    "estimator": "estimators",
    "scenario": "scenarios",
    "metric": "metrics",
    "report": "reports",
}

def register_registry_commands(app: typer.Typer) -> None:
    @app.command("show")
    def show_cmd(
        kind: str = typer.Argument(..., help="Kind: estimator(s)|scenario(s)|metric(s)|report(s)"),
        item_id: str = typer.Argument(..., help="Registered ID"),
    ):
        _, get_meta = lazy_registry()
        kind_norm = SINGULAR_TO_PLURAL.get(kind, kind)  # accept singular or plural
        try:
            meta = get_meta(kind_norm, item_id)
        except ValueError:
            # Give a friendly hint if a singular was used
            hint = SINGULAR_TO_PLURAL.get(kind, None)
            if hint:
                echo_err(f"Unknown kind '{kind}'. Did you mean '{hint}'?")
            raise
        if not meta:
            echo_err(f"{kind_norm[:-1]} '{item_id}' not found.")  # strip trailing 's' for message
        typer.echo(json.dumps(meta, indent=2))