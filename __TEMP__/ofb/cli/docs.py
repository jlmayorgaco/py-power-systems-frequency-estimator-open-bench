from __future__ import annotations

from pathlib import Path
import json
import typer

app = typer.Typer(add_completion=False, help="OpenFreqBench docs helpers")

def _require(obj, name: str):
    if obj is None:
        typer.secho(f"Missing subsystem: {name}", fg=typer.colors.RED)
        raise typer.Exit(1)
    return obj

def _imp():
    try:
        from ofb.core.registry import list_items, get_meta
    except Exception:
        list_items = None  # type: ignore
        get_meta = None  # type: ignore
    return list_items, get_meta

@app.command("build")
def build_docs(
    out_md: Path = typer.Option(Path("docs/SUITE.md"), "--out"),
):
    """(Re)build docs/SUITE.md with current registry snapshot."""
    list_items, get_meta = _imp()
    list_items = _require(list_items, "core.registry.list_items")
    get_meta = _require(get_meta, "core.registry.get_meta")

    kinds = ["estimators", "scenarios", "metrics", "reports"]
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# OpenFreqBench Suite\n"]
    for k in kinds:
        lines.append(f"\n## {k.title()}\n")
        items = list_items(k)
        if not items:
            lines.append("_none registered_\n")
            continue
        for it in items:
            mid = it["id"]
            meta = get_meta(k[:-1], mid) or {}
            desc = meta.get("description", "")
            lines.append(f"- `{mid}` — {desc}\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    typer.secho(f"Wrote {out_md}", fg=typer.colors.GREEN)


@app.command("suite")
def dump_suite_json(
    out_json: Path = typer.Option(Path("docs/artifacts/suite.json"), "--out"),
):
    """Emit a machine-readable snapshot of the registry."""
    list_items, _ = _imp()
    list_items = _require(list_items, "core.registry.list_items")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimators": list_items("estimators"),
        "scenarios": list_items("scenarios"),
        "metrics": list_items("metrics"),
        "reports": list_items("reports"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.secho(f"Wrote {out_json}", fg=typer.colors.GREEN)


def main():
    app()


if __name__ == "__main__":
    main()
