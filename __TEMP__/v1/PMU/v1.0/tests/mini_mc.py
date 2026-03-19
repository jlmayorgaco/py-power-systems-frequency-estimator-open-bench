#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/mini_mc.py — tiny Monte Carlo smoke-test
Goal: run 1 scenario, few methods, oracle+deployable, ~3 runs
and export everything (cfg snapshot, grids snapshot, plan, tuning, metrics).

Repo layout assumed (your tree):
  experiments/config.py
  experiments/mc.py
  experiments/build_registry.py
  experiments/io.py  (or project_io/*) for signals catalog
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Callable, Optional

from estimators.kalman import RAEKF2Estimator


# ----------------------------
# Helpers: JSON IO
# ----------------------------
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def _normalize_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      A) flat config: { fs_dsp_hz, mc, tuners, methods, ... }
      B) wrapper:
         { "experiment": { ... }, "tuners": {...}, "registry": {...} }
         or { "experiment": { "experiment": {...}, ... }, ... }
    Return a flat dict suitable for ExperimentConfig(**d).
    """
    if "fs_dsp_hz" in raw and "mc" in raw:
        return raw

    out: Dict[str, Any] = {}
    exp = raw.get("experiment", {})

    if isinstance(exp, dict) and "fs_dsp_hz" in exp:
        out.update(exp)
    elif (
        isinstance(exp, dict)
        and "experiment" in exp
        and isinstance(exp["experiment"], dict)
    ):
        out.update(exp["experiment"])

    if "tuners" in raw:
        out["tuners"] = raw["tuners"]
    if "registry" in raw:
        out["registry"] = raw["registry"]

    return out


# ----------------------------
# Helpers: Param grids (REAL names)
# ----------------------------
def _mini_param_grids() -> Dict[str, Any]:
    """
    Param-grid using REAL param names used across your pipeline:
      - KF:  kf_q, kf_r
      - PLL: pll_kp, pll_ki
      - RLS: rls_lam, rls_win
    """
    return {
        "kf_q": [1e-8, 1e-7, 1e-6, 1e-5],
        "kf_r": [1e-4, 1e-3, 1e-2, 1e-1],
        "pll_kp": [0.5, 1.0, 2.0, 5.0],
        "pll_ki": [1.0, 5.0, 10.0, 50.0, 100.0],
        "rls_lam": [0.98, 0.99, 0.995, 0.998],
        "rls_win": [10, 25, 50],
    }


def _mini_cfg(
    scenario_id: str,
    methods: List[str],
    n_test_runs: int,
    base_seed: int,
    export_dir_tag: str,
) -> Dict[str, Any]:
    """
    Build a SMALL flat cfg. IMPORTANT:
      - Do NOT include downsampling_ratio (ExperimentConfig doesn't accept it in your repo).
    """
    return {
        "fs_physics_hz": 100000,
        "fs_dsp_hz": 10000,
        "seed": 42,
        "methods": methods,
        "enable_landscapes": False,
        "enable_global_summaries": True,
        # optional: keep if your ExperimentConfig supports it; we will filter unknown keys anyway
        "scenario": {"T": 5.0},
        # Use param-grid tuners to validate tuning path
        "tuners": _mini_param_grids(),
        "mc": {
            "snr_db": 40.0,
            "n_train_seeds": 1,
            "n_test_seeds": int(n_test_runs),
            "enable_oracle_mode": True,
            "enable_deployable_mode": True,
            "base_seed": int(base_seed),
            "tune_frac": 0.2,
            "tune_frac_global": 0.1,
            "tune_frac_oracle": 1.0,
            "enable_sensitivity": False,
            "report_percentiles": [50, 99],
            "perturb": {
                "amp_pct_jitter": 0.02,
                "snr_db_jitter": 1.0,
                "impulsive_prob": 0.002,
                "impulsive_scale": 5.0,
                "rocof_pct_jitter": 0.1,
                "harmonics_pct_jitter": 0.05,
                "timing_samples_jitter": 1,
            },
            "force_serial_eval": True,
            "n_workers": 1,
            "split_test_frac": 0.5,
            "export_waveforms_modes": ["deployable", "oracle"],
            "train_scenarios": [scenario_id],
            "test_scenarios": [scenario_id],
        },
        "tag": f"mini_mc::{export_dir_tag}",
    }


# ----------------------------
# Helpers: filter unknown keys for dataclasses
# ----------------------------
def _filter_kwargs_for_dataclass(cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove keys not accepted by dataclass cls (top-level only).
    Prints what was removed so debugging is transparent.
    """
    if not is_dataclass(cls):
        return d  # can't filter reliably

    allowed = {f.name for f in fields(cls)}
    removed = sorted([k for k in d.keys() if k not in allowed])
    if removed:
        print("[mini_mc] stripping unknown ExperimentConfig keys:", removed)
    return {k: v for k, v in d.items() if k in allowed}


# ----------------------------
# Helpers: locate signals catalog function
# ----------------------------
def _resolve_signals_catalog_fn() -> Callable[[], Any]:
    """
    Try to find a callable that returns the signals/scenario catalog.
    You can adapt the candidates if your repo uses different naming.
    """
    candidates: List[tuple[str, str]] = [
        ("experiments.io", "get_signals_catalog"),
        ("experiments.io", "load_signals_catalog"),
        ("experiments.io", "build_signals_catalog"),
        ("project_io", "get_signals_catalog"),
        ("project_io", "load_signals_catalog"),
    ]

    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue

    raise RuntimeError(
        "Could not find a signals catalog function.\n"
        "Search in your repo for something like 'get_signals_catalog' or 'signals_catalog'.\n"
        "Quick command:\n"
        '  rg -n "signals_catalog|get_signals|catalog" experiments project_io\n'
    )


# ----------------------------
# Main
# ----------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        type=str,
        default="SIN_G1_E1",
        help="scenario_id string (must exist in your signals catalog)",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="RA-EKF2,MAF-SRF-PLL,RLS,DDSRF-PLL",
        help="comma-separated method list",
    )
    ap.add_argument("--runs", type=int, default=3, help="n_test_seeds (debug runs)")
    ap.add_argument("--base-seed", type=int, default=777, help="base seed")
    ap.add_argument(
        "--from-json",
        type=str,
        default="",
        help="Optional: load an existing config JSON then override with mini settings",
    )
    ap.add_argument(
        "--write-cfg",
        type=str,
        default="artifacts/tests/mini_mc_cfg.json",
        help="Where to write the mini cfg snapshot (input to mc.py too)",
    )
    args = ap.parse_args(argv[1:])

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise RuntimeError("Empty --methods")

    # Load base cfg (optional)
    if args.from_json:
        raw = _load_json(args.from_json)
        cfg_dict = _normalize_cfg(raw)
    else:
        cfg_dict = {}

    # Override with mini settings
    mini = _mini_cfg(
        scenario_id=str(args.scenario),
        methods=methods,
        n_test_runs=int(args.runs),
        base_seed=int(args.base_seed),
        export_dir_tag=f"{args.scenario}::{'-'.join(methods)}",
    )

    merged = copy.deepcopy(cfg_dict)
    merged.update(mini)

    # Ensure tuners exist
    if not isinstance(merged.get("tuners", None), dict) or len(merged["tuners"]) == 0:
        merged["tuners"] = _mini_param_grids()

    # Save cfg snapshot
    _dump_json(args.write_cfg, merged)
    print(f"[mini_mc] wrote cfg -> {args.write_cfg}")

    # Imports from your repo structure
    try:
        from experiments.config import ExperimentConfig
        from experiments.mc import MonteCarloRunner
        from experiments.build_registry import build_registry
    except Exception as e:
        raise RuntimeError(
            "Could not import experiments modules. Run from repo root so 'experiments/' is importable.\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    # Filter unknown top-level keys (prevents 'unexpected keyword' crashes)
    merged2 = _filter_kwargs_for_dataclass(ExperimentConfig, merged)

    # Construct cfg + registry + signals catalog
    cfg = ExperimentConfig(**merged2)
    registry = build_registry(cfg)

    signals_fn = _resolve_signals_catalog_fn()
    signals = signals_fn()

    # Run
    runner = MonteCarloRunner(cfg, registry)
    runner.run(signals)

    print(
        "[mini_mc] done. Check artifacts/ for exports (cfg snapshot, tuning, metrics, waveforms)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
