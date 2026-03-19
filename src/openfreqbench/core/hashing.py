"""Deterministic hashing utilities for artifact identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def dict_hash(d: dict[str, Any], length: int = 8) -> str:
    """SHA-256 hash of a JSON-serialised dict, truncated to `length` hex chars."""
    raw = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:length]
