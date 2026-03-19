from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass(frozen=True)
class DefaultGrids:
    ipdft_cycles: List[int] = (2, 3, 4, 6, 8, 10)
    ipdft_decim: List[int] = (1, 10, 50, 100)

    pll_kp: int = 20
    pll_ki: int = 30
    pll_kp_range: tuple = (1.0, 60.0)
    pll_ki_range: tuple = (1.0, 200.0)

    kf_q: List[float] = tuple(np.logspace(-2, 4, 8).tolist())
    kf_r: List[float] = tuple(np.logspace(-4, 1, 8).tolist())

    sogi_k: List[float] = (1.0, 1.414, 2.0)
    sogi_g: List[float] = (50.0, 100.0, 200.0, 300.0)

    rls_lam: List[float] = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)
    rls_win: List[int] = (50, 100, 200, 500)

    teager_win: List[int] = (10, 20, 30, 40, 50)
    tft_win: List[int] = (2, 3, 4, 6)

    vff_lam_min: List[float] = (0.90, 0.95, 0.98, 0.99)
    vff_ka: List[float] = (1.0, 2.0, 5.0)
    vff_win_smooth: int = 20
    vff_decim: int = 50

    koopman_win: List[int] = (10, 40, 80, 160, 333, 800, 2000, 5000)


def build_default_grids(
    cfg: DefaultGrids | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns method_name -> list of parameter dictionaries (grid).
    """
    cfg = cfg or DefaultGrids()

    kp_vals = np.linspace(cfg.pll_kp_range[0], cfg.pll_kp_range[1], cfg.pll_kp).tolist()
    ki_vals = np.linspace(cfg.pll_ki_range[0], cfg.pll_ki_range[1], cfg.pll_ki).tolist()

    return {
        "IpDFT": [
            {"cycles": int(c), "decim": int(d)}
            for c in cfg.ipdft_cycles
            for d in cfg.ipdft_decim
        ],
        "PLL": [{"kp": float(kp), "ki": float(ki)} for kp in kp_vals for ki in ki_vals],
        "EKF": [{"Q": float(q), "R": float(r)} for q in cfg.kf_q for r in cfg.kf_r],
        "UKF": [
            {"Q": float(q), "R": float(r), "smooth_win": 10}
            for q in cfg.kf_q
            for r in cfg.kf_r
        ],
        "LKF": [
            {"Q": float(q), "R": float(r), "smooth_win": 10}
            for q in cfg.kf_q
            for r in cfg.kf_r
        ],
        "SOGI": [
            {"k": float(k), "g": float(g)} for k in cfg.sogi_k for g in cfg.sogi_g
        ],
        "RLS": [
            {"lambda": float(lam), "win_smooth": int(w), "decim": 50}
            for lam in cfg.rls_lam
            for w in cfg.rls_win
        ],
        "Teager": [{"win": int(w)} for w in cfg.teager_win],
        "TFT": [{"win": int(w)} for w in cfg.tft_win],
        "RLS-VFF": [
            {
                "lam_min": float(lm),
                "lam_max": 0.9995,
                "Ka": float(ka),
                "Kb": None,
                "win_smooth": int(cfg.vff_win_smooth),
                "decim": int(cfg.vff_decim),
            }
            for lm in cfg.vff_lam_min
            for ka in cfg.vff_ka
        ],
        "Koopman-RKDPmu": [{"window_samples": int(w)} for w in cfg.koopman_win],
    }
