"""Canonical path layout helpers for the artifact store."""

from __future__ import annotations

from pathlib import Path


def artifacts_root(base: str | Path = "artifacts") -> Path:
    return Path(base)


def scenario_dir(base: str | Path, scenario_id: str) -> Path:
    return Path(base) / scenario_id


def method_dir(base: str | Path, scenario_id: str, method_id: str) -> Path:
    return Path(base) / scenario_id / method_id
