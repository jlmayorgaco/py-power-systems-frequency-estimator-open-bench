"""
openfreqbench/io/artifact_store.py

ArtifactStore: deterministic path layout for benchmark artifacts.

Layout
------
  <root>/<scenario_id>/<method_tag>/<method_tag>_<timestamp>_<artifact>.<ext>

ArtifactIdentity
----------------
Provides a deterministic cache key from (scenario_id, method_id, seed, params_hash).
This enables resume / skip-if-done logic without re-running completed work.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Deterministic identity / cache key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactIdentity:
    """Unique identity for one (scenario × method × seed) result."""

    scenario_id: str
    method_id: str
    seed: int
    scenario_params_hash: str   # sha256[:8] of scenario config dict
    method_params_hash: str     # sha256[:8] of method params dict
    schema_version: str = "v1"

    @classmethod
    def from_dicts(
        cls,
        scenario_id: str,
        method_id: str,
        seed: int,
        scenario_params: Dict[str, Any],
        method_params: Dict[str, Any],
        schema_version: str = "v1",
    ) -> "ArtifactIdentity":
        def _hash(d: Dict[str, Any]) -> str:
            raw = json.dumps(d, sort_keys=True, default=str).encode()
            return hashlib.sha256(raw).hexdigest()[:8]

        return cls(
            scenario_id=scenario_id,
            method_id=method_id,
            seed=seed,
            scenario_params_hash=_hash(scenario_params),
            method_params_hash=_hash(method_params),
            schema_version=schema_version,
        )

    @property
    def cache_key(self) -> str:
        parts = [
            self.schema_version,
            self.scenario_id,
            self.scenario_params_hash,
            self.method_id,
            self.method_params_hash,
            str(self.seed),
        ]
        return "__".join(parts)


# ---------------------------------------------------------------------------
# NaN-safe JSON encoder
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            if v != v:  # NaN
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


class ArtifactStore:
    """
    Manages artifact persistence under a root directory.

    Usage
    -----
        store = ArtifactStore("artifacts/")
        store.save_json(identity, "report", data_dict)
        store.save_csv(identity, "runs_long", df)
    """

    SCHEMA_VERSION = "v1"

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def _ts(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _dir(self, identity: ArtifactIdentity) -> Path:
        d = self.root / identity.scenario_id / identity.method_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _prefix(self, identity: ArtifactIdentity, ts: Optional[str] = None) -> str:
        ts = ts or self._ts()
        return f"{identity.method_id}_{ts}"

    def save_json(
        self,
        identity: ArtifactIdentity,
        artifact_name: str,
        data: Dict[str, Any],
        ts: Optional[str] = None,
    ) -> Path:
        ts = ts or self._ts()
        path = self._dir(identity) / f"{self._prefix(identity, ts)}_{artifact_name}.json"
        path.write_text(
            json.dumps(data, indent=2, cls=_NumpyEncoder, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def save_csv(
        self,
        identity: ArtifactIdentity,
        artifact_name: str,
        df: Any,
        ts: Optional[str] = None,
    ) -> Path:
        """Save a pandas DataFrame to CSV. Pandas must be available."""
        ts = ts or self._ts()
        path = self._dir(identity) / f"{self._prefix(identity, ts)}_{artifact_name}.csv"
        df.to_csv(path, index=False)
        return path

    def exists(self, identity: ArtifactIdentity, artifact_name: str) -> bool:
        """Return True if a JSON artifact with this identity already exists."""
        d = self._dir(identity)
        pattern = f"*_{artifact_name}.json"
        return any(d.glob(pattern))
