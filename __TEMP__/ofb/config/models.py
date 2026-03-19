from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal

class ScenarioCfg(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class EstimatorCfg(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class MetricCfg(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class ReportCfg(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class RuntimeCfg(BaseModel):
    mode: Literal["realtime","deterministic"] = "realtime"
    frame_period_s: float = Field(..., gt=0)
    duration_s: float = Field(..., gt=0)
    warmup_s: float = Field(0, ge=0)

class BenchmarkCfg(BaseModel):
    benchmark: Dict[str, Any]
    runtime: RuntimeCfg
    scenarios: List[ScenarioCfg]
    estimators: List[EstimatorCfg]
    metrics: List[MetricCfg] = []
    reports: List[ReportCfg] = []
