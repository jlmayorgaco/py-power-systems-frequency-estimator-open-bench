from __future__ import annotations
from typing import Any, Dict, Optional, List
import datetime
import hashlib
import os
import platform
import sys


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    experiment_cfg: Dict[str, Any],
    code_paths: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Manifest = what you cite in Q1 reproducibility paragraph.
    Stores:
      - full experiment config
      - environment metadata
      - optional code hashes for key files
    """
    code_paths = code_paths or []
    extra = extra or {}

    hashes: Dict[str, Optional[str]] = {p: _sha256_file(p) for p in code_paths}

    return {
        "created_at": str(datetime.datetime.now()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "experiment_cfg": experiment_cfg,
        "code_hashes_sha256": hashes,
        "extra": extra,
    }
