"""
openfreqbench/io/artifact_store.py

ArtifactStore + ArtifactIdentity for deterministic artifact layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True)
class ArtifactIdentity:
    scenario_id: str
    method_id: str
    seed: int
    scenario_params_hash: str
    method_params_hash: str
    schema_version: str = "v1"

    @classmethod
    def from_dicts(
        cls,
        scenario_id,
        method_id,
        seed,
        scenario_params,
        method_params,
        schema_version="v1",
    ):
        def _h(d):
            return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[
                :8
            ]

        return cls(
            scenario_id=scenario_id,
            method_id=method_id,
            seed=seed,
            scenario_params_hash=_h(scenario_params),
            method_params_hash=_h(method_params),
            schema_version=schema_version,
        )

    @property
    def cache_key(self) -> str:
        return "__".join(
            [
                self.schema_version,
                self.scenario_id,
                self.scenario_params_hash,
                self.method_id,
                self.method_params_hash,
                str(self.seed),
            ],
        )


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if v != v else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ArtifactStore:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def _dir(self, identity: ArtifactIdentity) -> Path:
        d = self.root / identity.scenario_id / identity.method_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _prefix(self, identity: ArtifactIdentity, ts: str | None = None) -> str:
        return f"{identity.method_id}_{ts or time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"

    def save_json(
        self,
        identity: ArtifactIdentity,
        artifact_name: str,
        data: dict[str, Any],
        ts: str | None = None,
    ) -> Path:
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
        ts: str | None = None,
    ) -> Path:
        path = self._dir(identity) / f"{self._prefix(identity, ts)}_{artifact_name}.csv"
        df.to_csv(path, index=False)
        return path

    def exists(self, identity: ArtifactIdentity, artifact_name: str) -> bool:
        return any(self._dir(identity).glob(f"*_{artifact_name}.json"))
