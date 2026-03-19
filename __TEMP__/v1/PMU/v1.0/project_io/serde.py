from __future__ import annotations
from typing import Any, Dict
import numpy as np


def numpy_to_builtin(x: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars to JSON-friendly Python objects.
    Keeps plain Python types untouched.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [numpy_to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {str(k): numpy_to_builtin(v) for k, v in x.items()}
    return x


def sanitize_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final sanitation pass before dumping to JSON.
    - Converts numpy types
    - Ensures keys are strings
    """
    out = numpy_to_builtin(payload)
    if not isinstance(out, dict):
        raise TypeError("sanitize_json expects a dict-like payload")
    return out
