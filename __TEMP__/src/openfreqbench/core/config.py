"""
openfreqbench/core/config.py

Pydantic v2 config models + YAML loader for benchmark.yaml.

Top-level structure:
  benchmark:
    name: my_run
    output_dir: artifacts
    n_runs: 100
  scenarios:
    - id: G1_E1_Pure_60Hz
      params: {}
  estimators:
    - id: ZeroCrossing
      params:
        filter_win: 5
  metrics:
    fs_hz: 10000
    f_nom: 60
    warm_up_s: 0.1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class BenchmarkMeta(BaseModel):
    name: str = "openfreqbench_run"
    output_dir: str = "artifacts"
    n_runs: int = Field(default=100, ge=1)
    seed_start: int = 0
    mode: int = Field(default=1, ge=1, le=5)


class ScenarioCfg(BaseModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EstimatorCfg(BaseModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MetricsCfg(BaseModel):
    fs_hz: float = 10_000.0
    f_nom: float = 60.0
    warm_up_s: float = 0.10
    ieee_fe_limit_mhz: float = 5.0
    ieee_rfe_limit_hzs: float = 0.1


class BenchmarkConfig(BaseModel):
    benchmark: BenchmarkMeta = Field(default_factory=BenchmarkMeta)
    scenarios: List[ScenarioCfg] = Field(default_factory=list)
    estimators: List[EstimatorCfg] = Field(default_factory=list)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark.yaml file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BenchmarkConfig.model_validate(raw or {})
