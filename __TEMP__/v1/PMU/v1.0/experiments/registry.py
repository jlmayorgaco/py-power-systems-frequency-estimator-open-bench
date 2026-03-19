# experiments/registry.py
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from estimators.factories import build_method_factories


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    builder: Callable[[Dict[str, Any]], Any]
    tuner: Optional[Callable[[], List[Dict[str, Any]]]] = None
    structural_latency: Optional[Callable[[Any], int]] = None


class MethodRegistry:
    def __init__(
        self, specs: Dict[str, MethodSpec], aliases: Optional[Dict[str, str]] = None
    ) -> None:
        self._specs = dict(specs)
        self._aliases = dict(aliases or {})

    def resolve(self, name: str) -> str:
        return self._aliases.get(name, name)

    def get(self, name: str) -> MethodSpec:
        key = self.resolve(name)
        if key not in self._specs:
            raise KeyError(
                f"[MethodRegistry] Unknown method '{name}'. Available: {', '.join(self.keys())}"
            )
        return self._specs[key]

    def keys(self) -> List[str]:
        return sorted(self._specs.keys())

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) in self._specs

    def __len__(self) -> int:
        return len(self._specs)


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def generate_grid(grid_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid_cfg:
        return []
    keys = sorted(grid_cfg.keys())
    values_lists: List[List[Any]] = []
    for k in keys:
        v = grid_cfg[k]
        values_lists.append(v if isinstance(v, list) else [v])
    return [dict(zip(keys, comb)) for comb in itertools.product(*values_lists)]


def _make_tuner_from_cfg(
    cfg: Dict[str, Any], tuner_key: Optional[str]
) -> Optional[Callable[[], List[Dict[str, Any]]]]:
    """
    Expects cfg["tuners"][tuner_key] = {param:[...], ...}
    """
    if not tuner_key:
        return None
    grid_data = _cfg_get(cfg, f"tuners.{tuner_key}", None)
    if grid_data is None:
        return None
    if not isinstance(grid_data, dict):
        raise TypeError(
            f"[registry] tuners.{tuner_key} must be a dict, got {type(grid_data)}"
        )
    return lambda: generate_grid(grid_data)


def _default_latency(est: Any) -> int:
    for attr in ("latency_samples", "window_samples", "N", "sz", "maf_win"):
        if hasattr(est, attr):
            try:
                v = int(getattr(est, attr))
                if v >= 0:
                    return v
            except Exception:
                pass
    return 0


def _make_safe_builder(
    factory: Callable[[Dict[str, Any]], Any], fixed_fs: float
) -> Callable[[Dict[str, Any]], Any]:
    def builder(p: Dict[str, Any]) -> Any:
        params = dict(p or {})
        params["fs_hz"] = float(fixed_fs)
        return factory(params)

    return builder


def build_registry(cfg: Dict[str, Any]) -> MethodRegistry:
    """
    Build the method registry used by BenchmarkRunner/MonteCarloRunner.

    Requirements:
      - cfg provides fs_dsp_hz (preferred) or fs_hz (fallback)
      - cfg may include cfg["tuners"] with grids per tuner_key
      - cfg["registry"]["aliases"] optional: { "PLL": "SRF-PLL", ... }
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

    METHOD_LIST: List[Dict[str, str]] = [
        {"name": "RA-EKF", "tuner": "ra-ekf", "family": "Kalman-Proposed"},
        {"name": "RA-EKF2", "tuner": "ra-ekf2", "family": "Kalman-SOTA"},
        {"name": "CKF", "tuner": "ckf", "family": "Kalman-Nonlinear"},
        {"name": "UKF", "tuner": "ukf", "family": "Kalman-Nonlinear"},
        {"name": "IEKF", "tuner": "iekf", "family": "Kalman-Nonlinear"},
        {"name": "EnKF", "tuner": "enkf", "family": "Kalman-Stochastic"},
        {"name": "EKF", "tuner": "ekf", "family": "Kalman-Classic"},
        {"name": "LKF", "tuner": "lkf", "family": "Kalman-Linear"},
        {"name": "IpDFT", "tuner": "ipdft", "family": "Fourier"},
        {"name": "TFT", "tuner": "tft", "family": "Time-Frequency"},
        {"name": "SOGI-Classic", "tuner": "sogi-classic", "family": "SOGI-FLL"},
        {"name": "SOGI-Industrial", "tuner": "sogi-industrial", "family": "SOGI-FLL"},
        {"name": "MSOGI-FLL", "tuner": "msogi-fll", "family": "SOGI-FLL"},
        {"name": "SRF-PLL", "tuner": "srf-pll", "family": "PLL"},
        {"name": "MAF-SRF-PLL", "tuner": "maf-srf-pll", "family": "PLL"},
        {"name": "DDSRF-PLL", "tuner": "ddsrf-pll", "family": "PLL"},
        {"name": "RLS", "tuner": "rls", "family": "Adaptive"},
        {"name": "RLS-VFF", "tuner": "rls-vff", "family": "Adaptive"},
        {"name": "Teager", "tuner": "teager", "family": "Energy-Op"},
    ]

    specs: Dict[str, MethodSpec] = {}

    for entry in METHOD_LIST:
        name = str(entry["name"])
        tuner_key = entry.get("tuner")
        family = str(entry.get("family", "Unknown"))

        if name not in factories:
            raise KeyError(
                f"[registry] Method '{name}' listed but not found in factories."
            )

        builder_func = _make_safe_builder(factories[name], fs_hz)

        specs[name] = MethodSpec(
            name=name,
            family=family,
            builder=builder_func,
            tuner=_make_tuner_from_cfg(cfg, tuner_key),
            structural_latency=(lambda est, _f=_default_latency: _f(est)),
        )

    aliases = _cfg_get(cfg, "registry.aliases", {}) or {}
    if not isinstance(aliases, dict):
        raise TypeError("[registry] registry.aliases must be a dict")

    return MethodRegistry(specs, aliases=aliases)
