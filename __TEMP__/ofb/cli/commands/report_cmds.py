from __future__ import annotations
from pathlib import Path
import typer

from ..libs.errors import echo_err
from ..libs.imports import lazy_reports


def register_report_commands(app: typer.Typer) -> None:
    @app.command("reports")
    def reports_cmd(
        results_dir: Path = typer.Argument(Path("results"), exists=False),
        report: str = typer.Option("standard", "--kind", "-k", help="Report kind"),
        outdir: Path = typer.Option(Path("docs/artifacts/reports"), "--outdir"),
    ):
        if report != "standard":
            echo_err("Only 'standard' is supported right now.")
        build_standard_report = lazy_reports()
        outdir.mkdir(parents=True, exist_ok=True)
        build_standard_report(results_dir, outdir)
        typer.secho(f"Report written to {outdir}", fg=typer.colors.GREEN)

    @app.command("calibrate")
    def calibrate_cmd(
        profile_secs: float = typer.Option(2.0, "--secs", help="How long to measure"),
    ):
        try:
            from ofb.runtime.profiling import MemoryMeter
            from ofb.runtime.timing import monotonic_ns
        except Exception:
            echo_err("runtime.profiling/timing not available yet.")
        meter = MemoryMeter()
        meter.start()
        start = monotonic_ns()
        x = 0
        while (monotonic_ns() - start) * 1e-9 < profile_secs:
            x += 1
        rss = meter.stop()
        typer.secho(
            f"Calibration: loop_iters={x:,}  rss≈{rss/1e6:.1f} MB",
            fg=typer.colors.GREEN,
        )
