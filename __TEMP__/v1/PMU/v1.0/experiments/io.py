# experiments/io.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np


def _is_np_scalar(x: Any) -> bool:
    return isinstance(x, (np.floating, np.integer, np.bool_))


def numpy_to_list(x: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    - np.ndarray -> list
    - np scalars -> float/int/bool
    - dict/list/tuple -> recursively converted
    - everything else -> returned as-is (must be JSON serializable upstream)
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [numpy_to_list(v) for v in x]
    if isinstance(x, dict):
        return {str(k): numpy_to_list(v) for k, v in x.items()}
    if _is_np_scalar(x):
        # preserve ints/bools cleanly
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        return float(x)
    return x


def _metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metrics dict is JSON safe:
    - numpy scalars -> python scalars
    - arrays/lists/dicts -> recursively converted
    """
    out: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if _is_np_scalar(v):
            if isinstance(v, np.bool_):
                out[str(k)] = bool(v)
            elif isinstance(v, np.integer):
                out[str(k)] = int(v)
            else:
                out[str(k)] = float(v)
        else:
            out[str(k)] = numpy_to_list(v)
    return out


def build_run_record(
    global_cfg: Dict[str, Any],
    scenario_name: str,
    scenario_meta: Dict[str, Any],
    method_name: str,
    method_family: str,
    method_tuning: Dict[str, Any],
    t: np.ndarray,
    v: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    metrics: Dict[str, Any],
    *,
    f_hat_raw: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build a single-run record payload (JSON-friendly).
    Supports both aligned output (f_hat) and optional raw output (f_hat_raw).
    """
    rec: Dict[str, Any] = {
        "metadata": {
            "global": numpy_to_list(global_cfg),
            "scenario": {
                "name": str(scenario_name),
                "meta": numpy_to_list(scenario_meta),
            },
            "method": {
                "name": str(method_name),
                "family": str(method_family),
                "tuning": numpy_to_list(method_tuning),
            },
        },
        "signals": {
            "t": numpy_to_list(t),
            "v": numpy_to_list(v),
            "f_true": numpy_to_list(f_true),
            # NOTE: f_hat is the aligned trace used for metrics (may contain NaNs at tail)
            "f_hat": numpy_to_list(f_hat),
        },
        "metrics": _metrics_to_json(metrics),
    }

    if f_hat_raw is not None:
        rec["signals"]["f_hat_raw"] = numpy_to_list(f_hat_raw)

    return rec


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Save payload to JSON with safe directory creation.
    """
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    # default=str prevents crashes on unexpected objects (shouldn't happen, but safer)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
