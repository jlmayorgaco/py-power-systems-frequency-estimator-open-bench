"""ofb scaffold — generate estimator/scenario/config stubs from Jinja templates."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from rich.console import Console
import typer

console = Console()

# Map kind → (template name, output subdir, class suffix)
_TEMPLATES = {
    "estimator": ("estimator.py.jinja", "src/openfreqbench/estimators/families", "Estimator"),
    "scenario": ("scenario.py.jinja", "src/openfreqbench/scenarios/g2", "Scenario"),
    "config": ("config.yaml.jinja", "examples", ""),
}


def scaffold_cmd(
    kind: Annotated[
        str, typer.Argument(help="What to scaffold: 'estimator', 'scenario', or 'config'"),
    ],
    name: Annotated[str, typer.Argument(help="CamelCase name, e.g. MyNewEstimator")],
    output_dir: Annotated[
        Path | None, typer.Option("--output", "-o", help="Override output directory."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print output path without writing."),
    ] = False,
) -> None:
    """Generate a stub file from a Jinja2 template."""
    kind = kind.lower()
    if kind not in _TEMPLATES:
        console.print(f"[red]Unknown kind {kind!r}.[/red] Use: {', '.join(_TEMPLATES)}")
        raise typer.Exit(code=1)

    template_name, default_out, suffix = _TEMPLATES[kind]

    # Find templates directory relative to this file
    templates_dir = Path(__file__).parent.parent / "templates"
    template_path = templates_dir / template_name

    if not template_path.exists():
        console.print(f"[red]Template not found:[/red] {template_path}")
        raise typer.Exit(code=1)

    try:
        from jinja2 import Template
    except ImportError as exc:
        console.print("[red]jinja2 not installed.[/red] Run: pip install jinja2")
        raise typer.Exit(code=1) from exc

    out_dir = output_dir or Path(default_out)
    snake_name = _to_snake(name)
    ext = ".yaml" if kind == "config" else ".py"
    out_path = out_dir / f"{snake_name}{ext}"

    rendered = Template(template_path.read_text(encoding="utf-8")).render(
        class_name=name,
        snake_name=snake_name,
        suffix=suffix,
    )

    if dry_run:
        console.print(f"[yellow]Dry-run:[/yellow] would write -> {out_path.resolve()}")
        console.print(rendered)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Created:[/green] {out_path.resolve()}")
    console.print(
        f"Next: implement [bold]{name}[/bold] and register it in [bold]core/registry.py[/bold]",
    )


def _to_snake(name: str) -> str:
    import re

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()
