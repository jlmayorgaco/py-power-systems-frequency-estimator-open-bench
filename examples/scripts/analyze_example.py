#!/usr/bin/env python3
"""Example: load and display results from a benchmark run."""
from __future__ import annotations

import json
from pathlib import Path

artifacts = Path("artifacts")
for report in sorted(artifacts.rglob("*_report.json")):
    data = json.loads(report.read_text())
    print(f"\n--- {report} ---")
    for k, v in data.get("aggregated", {}).items():
        print(f"  {k}: {v}")
