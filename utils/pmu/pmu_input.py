# utils/pmu/pmu_input.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- Data carriers ----------------------------------------------------------
@dataclass(slots=True)
class PMU_Input:
    V1: float
    V2: float
    V3: float
    I1: float
    I2: float
    I3: float
    timestamp: float  # high-precision UNIX UTC

    def validate(self) -> None:
        if not np.isfinite(
            [self.V1, self.V2, self.V3, self.I1, self.I2, self.I3, self.timestamp]
        ).all():
            raise ValueError("Non-finite value in input sample.")
