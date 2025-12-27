from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="OpenFreqBench scaffolding (Angular-style)")

# ----------------------------- helpers -----------------------------
RE_SNAKE = re.compile(r"^[a-z][a-z0-9_]+$")

VALID_ESTIMATOR_CATEGORIES = [
    "single",
    "multi",
    "distributed-single",
    "distributed-multi",
]


def _echo_err(msg: str, code: int = 1):
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_if_absent(path: Path, content: str):
    if path.exists():
        typer.secho(f"skip  {path} (exists)", fg=typer.colors.YELLOW)
        return
    path.write_text(content, encoding="utf-8")
    typer.secho(f"write {path}", fg=typer.colors.GREEN)


def _snake(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


# ----------------------------- estimator template -----------------------------
EST_SINGLE_TEMPLATE = """\
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ofb.core.estimators_api import single_estimator
from ofb.core.dto import Frame, EstimatorResult

@dataclass
class _State:
    x: float = 0.0
    p: float = 1.0

@single_estimator(
    id="{eid}",
    name="{ename}",
    category="{cat}",
    description="EKF toy example (replace with your research model)"
)
class Estimator:
    def __init__(self, fs: float = 5000.0):
        self.fs = fs
        self.state = _State()

    def reset(self):
        self.state = _State()

    def step(self, frame: Frame) -> EstimatorResult:
        # --- toy scalar EKF on phase angle derivative as frequency proxy ---
        y = frame.v  # assuming single-phase voltage sample
        # predict (identity)
        xp = self.state.x
        pp = self.state.p + 0.01
        # update
        H = 1.0
        R = 0.1
        K = pp * H / (H * pp * H + R)
        x = xp + K * (y - H * xp)
        p = (1 - K * H) * pp
        self.state.x, self.state.p = x, p

        # pretend `x` estimates instantaneous frequency deviation
        f_hat = 60.0 + 0.0 * x
        return EstimatorResult(
            f=f_hat,
            extra={{"debug_x": float(x), "K": float(K)}}
        )
"""

# ----------------------------- commands -----------------------------
@app.command("estimator")
def new_estimator(
    name: str = typer.Argument(..., help="Estimator package name (snake_case)"),
    category: str = typer.Option(
        ...,
        "--category",
        "-c",
        help="Estimator category",
        prompt=True,
    ),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Human title"),
    base_dir: Path = typer.Option(Path("ofb/estimators"), "--base-dir"),
):
    """
    Create a new estimator skeleton under `ofb/estimators/...`.
    Enforces category ∈ {single, multi, distributed-single, distributed-multi}.
    """
    if category not in VALID_ESTIMATOR_CATEGORIES:
        _echo_err(f"Invalid --category. Choose one of {VALID_ESTIMATOR_CATEGORIES}")

    pkg = _snake(name)
    if not RE_SNAKE.match(pkg):
        _echo_err("Name must be snake_case (e.g., ekf_single).")

    # layout: ofb/estimators/<family>/<name>/single.py
    # family from category prefix
    family = "state_space"  # you can adjust later (fft, pll, etc.)
    group = pkg  # keep group=name (users can restructure later)

    target_dir = base_dir / family / group
    _ensure_dir(target_dir)

    # __init__.py
    _write_if_absent(target_dir / "__init__.py", '"""Estimator package."""\n')

    # single.py (default entry)
    eid = f"{pkg}"
    ename = title or pkg.replace("_", " ").title()
    impl = EST_SINGLE_TEMPLATE.format(eid=eid, ename=ename, cat=category)
    _write_if_absent(target_dir / "single.py", impl)

    typer.secho(
        f"\nCreated estimator '{eid}' in {target_dir}/single.py\n"
        "Next steps:\n"
        "  1) implement your model in step()\n"
        "  2) add tests under tests/estimators/.../test_<name>.py\n"
        "  3) register in ofb.core.registry if not using decorators\n",
        fg=typer.colors.CYAN,
    )


@app.command("scenario")
def new_scenario(
    name: str = typer.Argument(..., help="Scenario module name (snake_case)"),
    base_dir: Path = typer.Option(Path("ofb/scenarios"), "--base-dir"),
):
    """Create a minimal scenario skeleton."""
    pkg = _snake(name)
    target = base_dir / pkg
    _ensure_dir(target)
    _write_if_absent(target / "__init__.py", '"""Scenario package."""\n')
    _write_if_absent(
        target / "example.py",
        """\
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ofb.scenarios.base import Scenario
from ofb.sources.synthetic import sine
from ofb.core.dto import Frame

@dataclass
class ExampleScenario(Scenario):
    fs: float = 5000.0
    duration: float = 2.0

    def build_source(self):
        # TODO: wire your source generator here
        t = np.arange(0, self.duration, 1.0/self.fs)
        v = np.sin(2*np.pi*60*t, dtype=np.float64)
        for sample in v:
            yield Frame(v=sample)
""",
    )
    typer.secho(f"Created scenario in {target}", fg=typer.colors.GREEN)


@app.command("metric")
def new_metric(
    name: str = typer.Argument(..., help="Metric module name (snake_case)"),
    base_dir: Path = typer.Option(Path("ofb/metrics"), "--base-dir"),
):
    """Create a metric skeleton."""
    pkg = _snake(name)
    target = base_dir / f"{pkg}.py"
    _ensure_dir(target.parent)
    _write_if_absent(
        target,
        """\
from __future__ import annotations
import numpy as np

def metric(values) -> float:
    \"\"\"Example metric (mean). Replace with IEEE 60255 FE/RFE, etc.\"\"\"
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr))
""",
    )
    typer.secho(f"Created metric at {target}", fg=typer.colors.GREEN)


@app.command("report")
def new_report(
    name: str = typer.Argument(..., help="Report module name (snake_case)"),
    base_dir: Path = typer.Option(Path("ofb/reports"), "--base-dir"),
):
    """Create a report skeleton."""
    pkg = _snake(name)
    target = base_dir / f"{pkg}.py"
    _ensure_dir(target.parent)
    _write_if_absent(
        target,
        """\
from __future__ import annotations
from pathlib import Path

def build(results_dir: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "README.txt").write_text("Custom report placeholder\\n", encoding="utf-8")
""",
    )
    typer.secho(f"Created report at {target}", fg=typer.colors.GREEN)


def main():
    app()


if __name__ == "__main__":
    main()
