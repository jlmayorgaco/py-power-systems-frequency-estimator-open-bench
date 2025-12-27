from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass(frozen=True)
class SingleIn: v: float

@dataclass(frozen=True)
class MultiIn:
    V3: Tuple[float, float, float]
    I3: Optional[Tuple[float, float, float]] = None
    V4: Optional[Tuple[float, float, float, float]] = None

@dataclass(frozen=True)
class DistributedSingleIn:
    v: float
    neighbors: Dict[str, float]

@dataclass(frozen=True)
class DistributedMultiIn:
    V3: Tuple[float, float, float]
    I3: Optional[Tuple[float, float, float]] = None
    neighbors: Dict[str, Dict[str, Tuple[float, ...]]] = None
