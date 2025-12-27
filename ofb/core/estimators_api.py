from dataclasses import dataclass
from typing import Any, Dict, Optional
from .registry import register

def estimator_single(name: str, *, category: str, meta: dict | None = None):
    def wrap(cls):
        m = {"category": category}; m.update(meta or {})
        register("estimator", f"estimators.{name}", meta=m)(cls); cls.name = name; return cls
    return wrap

def estimator_multi(name: str, *, category: str, meta: dict | None = None):
    def wrap(cls):
        m = {"category": category}; m.update(meta or {})
        register("estimator", f"estimators.{name}", meta=m)(cls); cls.name = name; return cls
    return wrap

def estimator_distributed(name: str, *, category: str, per_node: str, meta: dict | None = None):
    def wrap(cls):
        m = {"category": category, "per_node": per_node}; m.update(meta or {})
        register("estimator", f"estimators.{name}", meta=m)(cls); cls.name = name; cls.per_node = per_node; return cls
    return wrap

@dataclass
class EstimatorBase:  # shared bits
    memory: Dict[str, Any]
    def _result(self, *, t: float, seq: int, fhat: float, meta: Optional[Dict[str, Any]] = None):
        return {"t": t, "seq": seq, "fhat": fhat, "meta": meta or {}}

class EstimatorSingleBase(EstimatorBase):
    def init(self): ...
    def update(self, t: float, x): ...

class EstimatorMultiBase(EstimatorBase):
    def init(self): ...
    def update(self, t: float, x): ...

class EstimatorDistributedBase(EstimatorBase):
    def init(self): ...
    def update(self, t: float, x): ...
