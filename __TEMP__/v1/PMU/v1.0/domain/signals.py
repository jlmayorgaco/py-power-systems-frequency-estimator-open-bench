from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class Scenario:
    """
    Physical / simulation scenario definition.
    Metadata only (no signals).
    """

    name: str
    description: str
    params: Dict[str, Any]


@dataclass
class SignalBundle:
    """
    Container for time-series signals.
    """

    t: np.ndarray
    v: np.ndarray
    f_true: np.ndarray

    def __post_init__(self):
        n = len(self.t)
        assert len(self.v) == n, "v and t length mismatch"
        assert len(self.f_true) == n, "f_true and t length mismatch"

    @property
    def n_samples(self) -> int:
        return len(self.t)

    def slice(self, idx: np.ndarray) -> "SignalBundle":
        return SignalBundle(
            t=self.t[idx],
            v=self.v[idx],
            f_true=self.f_true[idx],
        )
