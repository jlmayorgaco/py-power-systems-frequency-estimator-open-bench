"""
openfreqbench/core/run_identity.py

RunIdentity: the canonical cache key for one (scenario × method × seed) run.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from openfreqbench.core.hashing import dict_hash


@dataclass(frozen=True)
class RunIdentity:
    """Deterministic identity for one benchmark trial."""
    scenario_id: str
    method_id: str
    seed: int
    scenario_params_hash: str
    method_params_hash: str
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
    ) -> "RunIdentity":
        return cls(
            scenario_id=scenario_id,
            method_id=method_id,
            seed=seed,
            scenario_params_hash=dict_hash(scenario_params),
            method_params_hash=dict_hash(method_params),
            schema_version=schema_version,
        )

    @property
    def cache_key(self) -> str:
        return "__".join([
            self.schema_version,
            self.scenario_id,
            self.scenario_params_hash,
            self.method_id,
            self.method_params_hash,
            str(self.seed),
        ])
