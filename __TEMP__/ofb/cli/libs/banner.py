# ofb/cli/libs/banner.py
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich import box
from typing import Optional

def build_banner(console: Console, *, project: str, version: str, author: str, max_width: Optional[int] = None) -> Panel:
    term_w = console.width
    w = min(max_width or term_w, term_w)
    title_txt = Text(f" ⚡ {project} ", style="bold white on blue", justify="center")
    tagline = Text.from_markup(
        "[bold cyan]Computationally-aware benchmarking suite for frequency and RoCoF "
        "estimators in low-inertia grids[/]"
    )
    meta = Table.grid(padding=(0, 3))
    meta.add_column(style="bold yellow", justify="right", no_wrap=True)
    meta.add_column(style="white")
    meta.add_row("🧭 Version:", f"[bold]{version}[/]")
    meta.add_row("👤 Author:", f"[bold]{author}[/]")
    meta.add_row("📂 Repo:", "[link=https://github.com/IngJorgeLuisMayorga/py-power-systems-frequency-estimator-open-bench]"
                              "GitHub • py-power-systems-frequency-estimator-open-bench[/link]")
    meta.add_row("📄 License:", "MIT")
    meta.add_row("🌍 Docs:", "Generate with `ofb-docs build` → docs/SUITE.md")

    quick = Table.grid(padding=(0, 3))
    quick.add_column(justify="right", style="bold green", no_wrap=True)
    quick.add_column()
    quick.add_row("▶️ Try:", "`ofb list estimators`")
    quick.add_row("🧪 Plan:", "`ofb plan benchmarks/configs/baseline_freq.yaml --expand-only`")
    quick.add_row("📊 Run:", "`ofb run-benchmark benchmarks/configs/baseline_freq.yaml`")
    quick.add_row("📑 Report:", "`ofb reports` → docs/artifacts/reports/")
    quick.add_row("🔧 Scaffold:", "`ofb-new estimator ekf_single -c single`")

    footer = Text.from_markup("[dim]IEEE/IEC-aligned benchmarking for modern PMUs • Built with Python[/dim]")

    body = Group(
        Align.center(title_txt),
        Rule(style="cyan"),
        Align.center(tagline),
        Text(),
        Align.center(meta, vertical="middle"),
        Text(),
        Rule(title="🚀 Quickstart", style="green"),
        Align.center(quick),
        Text(),
        Align.center(footer),
    )
    return Panel(body, border_style="bright_blue", box=box.DOUBLE, padding=(1, 6), width=w)
