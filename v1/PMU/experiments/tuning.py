# experiments/tuning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    # Evaluate all candidates
    for p in grid:
        s = _safe_float(score_fn(p))
        scored.append((s, p))
        if s < best_score:
            best_score = s
            best_params = dict(p)

    # Edge case: empty grid -> try defaults
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


def build_grids(grids: GridConfig) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build per-method parameter grids.

    IMPORTANT: Keys MUST match exactly:
      - cfg.methods in JSON
      - MethodRegistry keys (registry.get(name))
      - This dict keys (for tuning lookup)
    """
    def _lst(x: Any) -> List[Any]:
        return list(x) if x is not None else []

    return {
        # Spectral / windowed
       "IpDFT": [
            {"cycles": c, "decim": d} 
            for c in _lst(getattr(grids, "ipdft_cycles", []))
            for d in _lst(getattr(grids, "ipdft_decim", []))
        ],

        # IMPORTANT: PLL grid key must be "PLL" (because JSON now uses "PLL")
        # Your registry should map "PLL" to the SRF-PLL estimator factory internally.
        "PLL": [{"kp": kp, "ki": ki}
                for kp in _lst(getattr(grids, "pll_kp", []))
                for ki in _lst(getattr(grids, "pll_ki", []))],

        # If you still want to tune these explicitly, only include them if cfg.methods uses these names.
        # (Otherwise they will never be used.)
        # "MAF-SRF-PLL": [{"kp": kp, "ki": ki, "maf_win": 50}
        #                 for kp in _lst(getattr(grids, "pll_kp", []))
        #                 for ki in _lst(getattr(grids, "pll_ki", []))],
        # "DDSRF-PLL": [{"kp": kp, "ki": ki, "dec_tau_s": 0.02}
        #               for kp in _lst(getattr(grids, "pll_kp", []))
        #               for ki in _lst(getattr(grids, "pll_ki", []))],

        # IMPORTANT: EKF grid key must be "EKF" (because JSON now uses "EKF")
        "EKF": [{"Q": q, "R": r}
                for q in _lst(getattr(grids, "kf_q", []))
                for r in _lst(getattr(grids, "kf_r", []))],

        # These are separate method names; keep only if cfg.methods includes them.
        "UKF": [{"Q": q, "R": r, "smooth_win": 10}
                for q in _lst(getattr(grids, "kf_q", []))
                for r in _lst(getattr(grids, "kf_r", []))],

        "LKF": [{"Q": q, "R": r, "smooth_win": 10}
                for q in _lst(getattr(grids, "kf_q", []))
                for r in _lst(getattr(grids, "kf_r", []))],

        # SOGI
        "SOGI": [{"k": k, "g": g}
                 for k in _lst(getattr(grids, "sogi_k", []))
                 for g in _lst(getattr(grids, "sogi_g", []))],

        # RLS-like
        "RLS": [{"lambda": lam, "win_smooth": w, "decim": 50}
                for lam in _lst(getattr(grids, "rls_lam", []))
                for w in _lst(getattr(grids, "rls_win", []))],

        "Teager": [{"win": w} for w in _lst(getattr(grids, "teager_win", []))],
        "TFT": [{"win": w} for w in _lst(getattr(grids, "tft_win", []))],

        "RLS-VFF": [{
            "lam_min": lm,
            "lam_max": 0.9995,
            "Ka": ka,
            "Kb": None,
            "win_smooth": getattr(grids, "vff_win_smooth", 1),
            "decim": getattr(grids, "vff_decim", 1),
        } for lm in _lst(getattr(grids, "vff_lam_min", []))
          for ka in _lst(getattr(grids, "vff_ka", []))],

        "Koopman-RKDPmu": [{"window_samples": w} for w in _lst(getattr(grids, "koopman_win", []))],
        # EKF2 and PI-GRU intentionally absent (fixed / tuned elsewhere)
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
    """
    Generic grid-search tuner.

    base_params: merged into each candidate with candidate taking precedence.

    Safety:
      - Resets estimator if it has .reset()
      - Guards exceptions and NaNs so tuning doesn't crash
    """
    _ = method_name  # keep for future logging
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
