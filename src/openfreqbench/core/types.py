"""Shared type aliases used across the package."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

Array = np.ndarray
ParamsDict = Dict[str, Any]
MetricDict = Dict[str, Any]
AggDict = Dict[str, Dict[str, float]]
