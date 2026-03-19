# project_io/config.py
from __future__ import annotations
import json
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
