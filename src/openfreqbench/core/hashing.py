"""Deterministic hashing utilities for artifact identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted([_normalize(x) for x in obj])
    return obj

def dict_hash(d: dict[str, Any], length: int = 8) -> str:
    """SHA-256 hash of a JSON-serialised dict, truncated to `length` hex chars."""
    raw = json.dumps(_normalize(d), default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:length]
