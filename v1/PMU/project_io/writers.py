from __future__ import annotations
from typing import Any, Dict, Optional
import json
import os

from .serde import sanitize_json


class JsonWriter:
    """
    Single responsibility: write JSON payloads deterministically.
    """

    def __init__(self, indent: int = 2, sort_keys: bool = False) -> None:
        self.indent = indent
        self.sort_keys = sort_keys

    def write(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sanitize_json(payload), f, indent=self.indent, sort_keys=self.sort_keys)

    def write_compact(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sanitize_json(payload), f, separators=(",", ":"), sort_keys=self.sort_keys)
