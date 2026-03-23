"""
openfreqbench/core/config_models.py

Pydantic v2 config models + YAML loader for benchmark.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
import yaml


class BenchmarkMeta(BaseModel):
    name: str = "openfreqbench_run"
    output_dir: str = "artifacts"
    n_runs: int = Field(default=100, ge=1)
    seed_start: int = 0
    mode: int = Field(default=1, ge=1, le=5)
    plots: bool = False
    n_tune_eval: int = Field(default=5, ge=1)  # seeds used during grid-search


class MCPerturbSpec(BaseModel):
    """Describes a Monte Carlo perturbation for one scenario parameter."""

    mode: Literal["rel", "abs"] = "rel"  # rel = fraction of nominal, abs = additive
    scale: float = 0.0  # std-dev of the perturbation

    model_config = {"extra": "forbid"}


class ScenarioCfg(BaseModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict)
    mc_perturbations: dict[str, MCPerturbSpec] = Field(default_factory=dict)


class EstimatorCfg(BaseModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict)


class MetricsCfg(BaseModel):
    fs_hz: float = 10_000.0
    f_nom: float = 60.0
    warm_up_s: float = 0.10
    ieee_fe_limit_mhz: float = 5.0
    ieee_rfe_limit_hzs: float = 0.1


class BenchmarkConfig(BaseModel):
    benchmark: BenchmarkMeta = Field(default_factory=BenchmarkMeta)
    scenarios: list[ScenarioCfg] = Field(default_factory=list)
    estimators: list[EstimatorCfg] = Field(default_factory=list)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)


def load_config(path: str | Path) -> BenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BenchmarkConfig.model_validate(raw or {})
