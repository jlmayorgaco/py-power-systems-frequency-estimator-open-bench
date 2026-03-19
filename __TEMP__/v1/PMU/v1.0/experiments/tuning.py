# experiments/tuning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import product

import numpy as np

from experiments.config import GridConfig


@dataclass(frozen=True)
class TuningResult:
    label: str
    params: Dict[str, Any]
    best_score: float
    n_evals: int


def _label_from_params(p: Dict[str, Any]) -> str:
    if not p:
        return "default"
    parts: List[str] = []
    for k in sorted(p.keys()):
        if k == "label":
            continue
        parts.append(f"{k}={p[k]}")
    return ",".join(parts) if parts else "default"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    return v if np.isfinite(v) else float("inf")


def _grid_search(
    score_fn: Callable[[Dict[str, Any]], float],
    grid: List[Dict[str, Any]],
    *,
    debug: bool = False,
    debug_topk: int = 10,
) -> TuningResult:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    best_params: Dict[str, Any] = {}
    best_score = float("inf")

    for p in grid:
        s = _safe_float(score_fn(p))
        scored.append((s, p))
        if s < best_score:
            best_score = s
            best_params = dict(p)

    if len(grid) == 0:
        best_score = _safe_float(score_fn({}))
        best_params = {}

    if debug:
        scored_sorted = sorted(scored, key=lambda x: x[0])
        print(f"[DBG][TUNING] Grid search results ({len(grid)} candidates):")
        for s, p in scored_sorted[: max(1, int(debug_topk))]:
            print(f"   {_label_from_params(p):<30} score={s:.6f}")
        print(f"[DBG][TUNING] Best score = {best_score:.6f}")
        print("-" * 60)

    return TuningResult(
        label=_label_from_params(best_params),
        params=best_params,
        best_score=float(best_score),
        n_evals=int(len(grid)),
    )


def build_grids(grids: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build per-method parameter grids.

    Supports:
      A) GridConfig dataclass
      B) JSON dict-of-dicts or list-of-dicts
    """
    if isinstance(grids, dict):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for method_key, spec in grids.items():
            if spec is None:
                out[method_key] = []
                continue
            if isinstance(spec, list):
                out[method_key] = [dict(x) for x in spec if isinstance(x, dict)]
                continue
            if not isinstance(spec, dict):
                out[method_key] = []
                continue

            keys = list(spec.keys())
            values_lists: List[List[Any]] = []
            for k in keys:
                v = spec[k]
                if isinstance(v, np.ndarray):
                    vv = list(np.ravel(v).tolist())
                elif isinstance(v, (list, tuple)):
                    vv = list(v)
                else:
                    vv = [v]
                if len(vv) == 0:
                    values_lists = []
                    break
                values_lists.append(vv)

            if not values_lists:
                out[method_key] = []
                continue

            combos: List[Dict[str, Any]] = []
            for vals in product(*values_lists):
                combos.append({k: vals[i] for i, k in enumerate(keys)})
            out[method_key] = combos
        return out

    def _lst(x: Any) -> List[Any]:
        return list(x) if x is not None else []

    # Legacy GridConfig
    g: GridConfig = grids

    rls_vff = []
    for lm in _lst(getattr(g, "vff_lam_min", [])):
        for ka in _lst(getattr(g, "vff_ka", [])):
            for ws in _lst(getattr(g, "vff_win_smooth", [])):
                for d in _lst(getattr(g, "vff_decim", [])):
                    rls_vff.append(
                        {
                            "lam_min": lm,
                            "lam_max": 0.9995,
                            "Ka": ka,
                            "Kb": None,
                            "win_smooth": ws,
                            "decim": d,
                        }
                    )

    return {
        "IpDFT": [
            {"cycles": c, "decim": d}
            for c in _lst(g.ipdft_cycles)
            for d in _lst(g.ipdft_decim)
        ],
        "PLL": [{"kp": kp, "ki": ki} for kp in _lst(g.pll_kp) for ki in _lst(g.pll_ki)],
        "EKF": [{"Q": q, "R": r} for q in _lst(g.kf_q) for r in _lst(g.kf_r)],
        "UKF": [
            {"Q": q, "R": r, "smooth_win": 10}
            for q in _lst(g.kf_q)
            for r in _lst(g.kf_r)
        ],
        "LKF": [
            {"Q": q, "R": r, "smooth_win": 10}
            for q in _lst(g.kf_q)
            for r in _lst(g.kf_r)
        ],
        "SOGI": [{"k": k, "g": gg} for k in _lst(g.sogi_k) for gg in _lst(g.sogi_g)],
        "RLS": [
            {"lambda": lam, "win_smooth": w, "decim": 50}
            for lam in _lst(g.rls_lam)
            for w in _lst(g.rls_win)
        ],
        "Teager": [{"win": w} for w in _lst(g.teager_win)],
        "TFT": [{"win": w} for w in _lst(g.tft_win)],
        "RLS-VFF": rls_vff,
        "Koopman-RKDPmu": [{"window_samples": w} for w in _lst(g.koopman_win)],
    }


def tune_generic(
    method_name: str,
    grid: List[Dict[str, Any]],
    make_estimator: Callable[[Dict[str, Any]], Any],
    score_from_trace: Callable[[np.ndarray], float],
    v_train: np.ndarray,
    *,
    base_params: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    debug_topk: int = 10,
) -> TuningResult:
    _ = method_name
    base_params = dict(base_params or {})
    v_train = np.asarray(v_train, dtype=float)

    def score_fn(p: Dict[str, Any]) -> float:
        params = {**base_params, **dict(p)}
        try:
            est = make_estimator(params)
            if hasattr(est, "reset") and callable(getattr(est, "reset")):
                try:
                    est.reset()
                except Exception:
                    pass

            tr = np.array([est.step(float(x)) for x in v_train], dtype=float)
            s = score_from_trace(tr)
            return _safe_float(s)
        except Exception:
            return float("inf")

    return _grid_search(score_fn, grid, debug=debug, debug_topk=debug_topk)
