# viz/summary.py
from __future__ import annotations
from typing import Any, Dict
import os
import json

from .tables import build_metrics_table, build_mc_table


def generate_global_summaries(
    out_dir: str,
    benchmark_json_path: str | None = None,
    mc_json_path: str | None = None,
) -> None:
    """
    Generate global CSV summaries (paper tables) from saved JSON outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    if benchmark_json_path and os.path.exists(benchmark_json_path):
        with open(benchmark_json_path, "r") as f:
            bench = json.load(f)
        df = build_metrics_table(bench)
        df.to_csv(os.path.join(out_dir, "benchmark_metrics_table.csv"), index=False)

    if mc_json_path and os.path.exists(mc_json_path):
        with open(mc_json_path, "r") as f:
            mc = json.load(f)
        df = build_mc_table(mc)
        df.to_csv(os.path.join(out_dir, "mc_metrics_table.csv"), index=False)
