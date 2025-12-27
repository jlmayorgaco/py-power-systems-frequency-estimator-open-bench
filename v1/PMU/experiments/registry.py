# experiments/registry.py
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from estimators.factories import build_method_factories


# ============================================================
# Public types
# ============================================================

@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    builder: Callable[[Dict[str, Any]], Any]
    tuner: Optional[Callable[[], List[Dict[str, Any]]]] = None
    structural_latency: Optional[Callable[[Any], int]] = None


class MethodRegistry:
    def __init__(self, specs: Dict[str, MethodSpec]) -> None:
        self._specs = dict(specs)

    def get(self, name: str) -> MethodSpec:
        if name not in self._specs:
            raise KeyError(f"[MethodRegistry] Unknown method '{name}'. Available: {', '.join(self.keys())}")
        return self._specs[name]

    def keys(self) -> List[str]:
        return sorted(self._specs.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


# ============================================================
# Helpers
# ============================================================

def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safe dotted-path access: _cfg_get(cfg, "tuners.ipdft", None)
    """
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _infer_family_from_instance(est: Any, fallback: str) -> str:
    fam = getattr(est, "FAMILY", None)
    if isinstance(fam, str) and fam.strip():
        return fam.strip()
    return fallback


def _make_safe_builder(fac_func: Callable[[Dict[str, Any]], Any], fixed_fs: float) -> Callable[[Dict[str, Any]], Any]:
    """
    Close over factory and fs to avoid late-binding bugs in loops.
    Always inject experiment fs_hz (DSP rate) as source of truth.
    """
    def builder(p: Dict[str, Any]) -> Any:
        params = dict(p or {})
        params["fs_hz"] = float(fixed_fs)
        return fac_func(params)
    return builder


# ============================================================
# Generic Grid Generator
# ============================================================

def generate_grid(grid_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turns {"a":[1,2], "b":[10,20]} -> [{"a":1,"b":10}, ...]
    Supports scalar values by wrapping into a list.
    """
    if not grid_cfg:
        return []
    keys = sorted(grid_cfg.keys())
    values_lists = []
    for k in keys:
        v = grid_cfg[k]
        if isinstance(v, list):
            values_lists.append(v)
        else:
            values_lists.append([v])
    return [dict(zip(keys, comb)) for comb in itertools.product(*values_lists)]


def _make_tuner(cfg: Dict[str, Any], tuner_key: Optional[str]) -> Optional[Callable[[], List[Dict[str, Any]]]]:
    """
    Expects cfg["tuners"][tuner_key] to be a dict of grid lists.
    Example: cfg["tuners"]["ipdft"] = {"cycles":[...], "decim":[...]}
    """
    if not tuner_key:
        return None
    grid_data = _cfg_get(cfg, f"tuners.{tuner_key}", None)
    if grid_data is None:
        return None
    if not isinstance(grid_data, dict):
        raise TypeError(f"[registry] tuners.{tuner_key} must be a dict, got {type(grid_data)}")
    return lambda: generate_grid(grid_data)


def _default_latency(est: Any) -> int:
    # Structural latency: try a few common fields
    for attr in ("latency_samples", "window_samples", "N", "sz", "maf_win"):
        if hasattr(est, attr):
            try:
                v = int(getattr(est, attr))
                if v >= 0:
                    return v
            except Exception:
                pass
    return 0


# ============================================================
# Build registry
# ============================================================

def build_registry(cfg: Dict[str, Any]) -> MethodRegistry:
    """
    Build the method registry used by BenchmarkRunner/MonteCarloRunner.

    Requirements:
      - cfg must include fs_dsp_hz (preferred) or fs_hz (fallback).
      - cfg may include cfg["tuners"] with grids per method family.
    """
    factories = build_method_factories()

    fs_hz = _cfg_get(cfg, "fs_dsp_hz", None)
    if fs_hz is None:
        fs_hz = _cfg_get(cfg, "fs_hz", None)
    if fs_hz is None:
        raise KeyError("[registry] cfg must provide 'fs_dsp_hz' (or 'fs_hz').")
    fs_hz = float(fs_hz)
    if fs_hz <= 0:
        raise ValueError(f"[registry] fs_hz must be > 0, got {fs_hz}")

    # ------------------------------------------------------------------
    # Master method list (registry names MUST match factories keys)
    # ------------------------------------------------------------------
    METHOD_LIST: List[Dict[str, str]] = [
        # --- Familia Kalman (El núcleo de tu investigación) ---
        {"name": "RA-EKF",          "tuner": "ra-ekf",          "family": "Kalman-Proposed"},
        {"name": "RA-EKF2",         "tuner": "ra-ekf2",         "family": "Kalman-SOTA"},
        {"name": "CKF",             "tuner": "ckf",             "family": "Kalman-Nonlinear"},
        {"name": "UKF",             "tuner": "ukf",             "family": "Kalman-Nonlinear"},
        {"name": "IEKF",            "tuner": "iekf",            "family": "Kalman-Nonlinear"},
        {"name": "EnKF",            "tuner": "enkf",            "family": "Kalman-Stochastic"},
        {"name": "EKF",             "tuner": "ekf",             "family": "Kalman-Classic"},
        {"name": "LKF",             "tuner": "lkf",             "family": "Kalman-Linear"},

        # --- Familia Fourier (Transformadas de Tiempo-Frecuencia) ---
        {"name": "IpDFT",           "tuner": "ipdft",           "family": "Fourier"},
        {"name": "TFT",             "tuner": "tft",             "family": "Time-Frequency"},

        # --- Familia SOGI-FLL (Estándares Monofásicos) ---
        {"name": "SOGI-Classic",    "tuner": "sogi-classic",    "family": "SOGI-FLL"},
        {"name": "SOGI-Industrial", "tuner": "sogi-industrial", "family": "SOGI-FLL"},
        {"name": "MSOGI-FLL",       "tuner": "msogi-fll",       "family": "SOGI-FLL"},

        # --- Familia Control / PLL (Comparativa de Lazo Cerrado) ---
        {"name": "SRF-PLL",         "tuner": "srf-pll",         "family": "PLL"},
        {"name": "MAF-SRF-PLL",     "tuner": "maf-srf-pll",     "family": "PLL"},
        {"name": "DDSRF-PLL",       "tuner": "ddsrf-pll",       "family": "PLL"},

        # --- Métodos Adaptativos y de Energía ---
        {"name": "RLS",             "tuner": "rls",             "family": "Adaptive"},
        {"name": "RLS-VFF",         "tuner": "rls-vff",         "family": "Adaptive"},
        {"name": "Teager",          "tuner": "teager",          "family": "Energy-Op"}
    ]

    specs: Dict[str, MethodSpec] = {}

    for entry in METHOD_LIST:
        name = str(entry["name"])
        tuner_key = entry.get("tuner")
        fam_fallback = str(entry.get("family", "Unknown"))

        # Fail-fast if declared method doesn't exist in factories
        if name not in factories:
            # This makes debugging easier than silently skipping.
            # If you *want* silent skip, change this to `continue`.
            raise KeyError(f"[registry] Method '{name}' listed in METHOD_LIST but not found in factories.")

        factory_name = factories[name]
        print(' ')
        print(' ')
        print(' factories')
        print(factories)
        print(' ')
        print(' factory_name ')
        print(factory_name)
        print(' ')
        builder_func = _make_safe_builder(factories[name], fs_hz)

        # Try to infer true family from estimator instance, but NEVER crash registry build.
        family = fam_fallback
        try:
            tmp_est = builder_func({})
            family = _infer_family_from_instance(tmp_est, fam_fallback)
        except Exception:
            family = fam_fallback

        specs[name] = MethodSpec(
            name=name,
            family=family,
            builder=builder_func,
            tuner=_make_tuner(cfg, tuner_key),
            structural_latency=(lambda est, _f=_default_latency: _f(est)),
        )

    return MethodRegistry(specs)
