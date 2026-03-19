#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Entry point (Q1 Research-Grade)

CRITICAL DESIGN RULE:
- DO NOT parallelize MonteCarloRunner per-scenario here.
  That breaks the MC plan (train/test scenario split) and yields plan.json with only 1 scenario.
- Let MonteCarloRunner (mc.py) handle parallelism internally per-run.

This version also FIXES:
- Registry/method-name mismatches via optional aliases + sane defaults
- Tuning grids: if config doesn't provide "tuners", we auto-build tuners from ExperimentConfig.grids
- Robust dataclass/dict handling (avoid ExperimentConfig TypeError)
- Manifest for traceability
- Single-mode runner for quick debugging

Usage:
  python ./main.py --mode mc --workers -1
  python ./main.py --mode single
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

# ---------------------------
# Infrastructure / IO
# ---------------------------
from project_io.config import load_config
from project_io.paths import ProjectPaths
from project_io.writers import JsonWriter
from project_io.manifest import build_manifest

# ---------------------------
# Domain / Experiment
# ---------------------------
from experiments import (
    ExperimentConfig,
    build_registry,
)

# ---------------------------
# Scenarios provider
# ---------------------------
# Your scenarios package should expose a function that returns a dict-like collection of signals
# e.g. {scenario_id: {"t":..., "v":..., "f_true":..., "meta":...}, ...}
from scenarios import get_test_signals


# ============================================================
# Utils
# ============================================================


def _sys_metadata() -> Dict[str, Any]:
    return {
        "timestamp": str(datetime.datetime.now()),
        "hostname": platform.node(),
        "machine_arch": platform.machine(),
        "cpu_processor": platform.processor(),
        "os_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "argv": sys.argv[:],
    }


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _asdict_safe(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


def _expect_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if is_dataclass(cfg):
        return asdict(cfg)
    try:
        return dict(cfg)
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--mode", type=str, default="mc", choices=["mc", "single", "figures"]
    )
    p.add_argument("--config", type=str, default="configs/mc_default.json")
    p.add_argument("--outdir", type=str, default="artifacts")
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Cores hint for MC internal parallelism (-1 auto). Passed via ENV if mc.py supports it.",
    )
    return p.parse_args()


def _extract_T(cfg_dict: Dict[str, Any], default_T: float = 5.0) -> float:
    """
    Try to read T from config; fallback to default.
    Keeps main stable if ExperimentConfig doesn't expose T.
    """
    try:
        exp = cfg_dict.get("experiment", {}) or {}
        if "T" in exp:
            return float(exp["T"])
        sc = exp.get("scenario", None)
        if isinstance(sc, dict) and "T" in sc:
            return float(sc["T"])
    except Exception:
        pass
    return float(default_T)


def _build_default_tuners_from_grids(exp_cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Build cfg["tuners"] dict in the exact "tuner_key -> {param:[...], ...}" format
    that experiments/registry.py expects (tuners.<tuner_key>).

    IMPORTANT:
    - GridConfig provides generic grids (kf_q, kf_r, pll_kp, pll_ki, ...)
    - We map them into your tuner keys used in METHOD_LIST:
        ra-ekf, ra-ekf2, ekf, ukf, lkf, ckf, iekf, enkf, ipdft, srf-pll, ...
    - If a method uses fixed params and you DON'T want tuning, leave it absent here.

    This is conservative: it enables tuning for the most standard knobs.
    """
    g = exp_cfg.grids

    tuners: Dict[str, Any] = {}

    # ---- Kalman-like (shared Q/R grids)
    # Use "Q" and "R" because your GridConfig legacy builder uses those keys;
    # your estimator factories must accept these keys (or adapt inside factories).
    kf_grid = {"Q": list(g.kf_q), "R": list(g.kf_r)}

    for key in ("ra-ekf", "ra-ekf2", "ekf", "ukf", "lkf", "ckf", "iekf", "enkf"):
        tuners[key] = dict(kf_grid)

    # ---- IpDFT
    tuners["ipdft"] = {
        "cycles": list(g.ipdft_cycles),
        "decim": list(g.ipdft_decim),
        # optional if your factory accepts it
        "window_type": list(g.ipdft_window_type),
    }

    # ---- PLL family
    pll_grid = {"kp": list(g.pll_kp), "ki": list(g.pll_ki)}
    for key in ("srf-pll", "maf-srf-pll", "ddsrf-pll"):
        tuners[key] = dict(pll_grid)

    # ---- SOGI family
    sogi_grid = {"k": list(g.sogi_k), "g": list(g.sogi_g)}
    for key in ("sogi-classic", "sogi-industrial", "msogi-fll"):
        tuners[key] = dict(sogi_grid)

    # ---- RLS
    tuners["rls"] = {
        "lambda": list(g.rls_lam),
        "win_smooth": list(g.rls_win),
        # if your factory accepts decim as a tunable knob, add it here.
        # "decim": [50],
    }

    # ---- RLS-VFF
    tuners["rls-vff"] = {
        "lam_min": list(g.vff_lam_min),
        "Ka": list(g.vff_ka),
        "win_smooth": list(g.vff_win_smooth),
        "decim": list(g.vff_decim),
        # keep lam_max fixed by default; factories can override
        "lam_max": [0.9995],
    }

    # ---- Teager
    tuners["teager"] = {"win": list(g.teager_win)}

    # ---- TFT
    tuners["tft"] = {"win": list(g.tft_win)}

    # ---- Koopman window (if you expose a tuner key for it; otherwise omit)
    # Example key; adjust if your registry uses something else.
    tuners["koopman-rkdpmu"] = {"window_samples": list(g.koopman_win)}

    return tuners


def _default_aliases() -> Dict[str, str]:
    """
    Backward-compat mapping: if older configs still say "PLL" or "SOGI" etc.
    Map them to registry keys.

    You can extend this freely; it only affects name resolution.
    """
    return {
        # legacy -> canonical
        "PLL": "SRF-PLL",
        "SOGI": "SOGI-Classic",
        "IpDFT": "IpDFT",
        "RLS": "RLS",
        "RLS-VFF": "RLS-VFF",
        "EKF2": "RA-EKF2",
        "EKF": "EKF",
        "UKF": "UKF",
        "LKF": "LKF",
        "Teager": "Teager",
        "TFT": "TFT",
        # common naming variants
        "SRF_PLL": "SRF-PLL",
        "MAF_SRF_PLL": "MAF-SRF-PLL",
        "DDSRF_PLL": "DDSRF-PLL",
    }


# ============================================================
# Main
# ============================================================


def main() -> int:
    args = parse_args()
    outdir = _ensure_dir(args.outdir)
    paths = ProjectPaths(base_dir=outdir)
    paths.ensure()
    writer = JsonWriter()

    # 1) Load configuration
    cfg_raw = load_config(args.config)
    if not cfg_raw:
        print(f"[ERROR] No se pudo cargar {args.config}")
        return 1
    cfg_dict = _expect_dict(cfg_raw)

    # 2) Experiment config (strip metadata to avoid TypeError)
    exp_data = (cfg_dict.get("experiment", {}) or {}).copy()
    exp_data.pop("metadata", None)
    try:
        exp_cfg = ExperimentConfig(**exp_data)
    except TypeError as e:
        print(f"[ERROR] ExperimentConfig kwargs inválidos: {e}")
        print("Keys recibidas:", sorted(list(exp_data.keys())))
        return 1

    # 3) Registry config injection
    registry_cfg: Dict[str, Any] = _expect_dict(cfg_dict.get("registry", {}))
    registry_cfg["fs_dsp_hz"] = float(exp_cfg.fs_dsp_hz)

    # ---- Tuners:
    # Prefer explicit cfg top-level "tuners". If missing/empty, auto-build from exp_cfg.grids.
    tuners_cfg = cfg_dict.get("tuners", None)
    if not isinstance(tuners_cfg, dict) or len(tuners_cfg) == 0:
        tuners_cfg = _build_default_tuners_from_grids(exp_cfg)
    registry_cfg["tuners"] = tuners_cfg

    # ---- Aliases for method names (optional but strongly recommended)
    registry_cfg.setdefault("registry", {})
    aliases = registry_cfg["registry"].get("aliases", None)
    if not isinstance(aliases, dict):
        aliases = {}
    # merge defaults without overwriting user-provided
    for k, v in _default_aliases().items():
        aliases.setdefault(k, v)
    registry_cfg["registry"]["aliases"] = aliases

    # 4) Optional: pass workers hint via env for mc.py
    if isinstance(args.workers, int) and args.workers != 0:
        os.environ["OFB_MC_WORKERS"] = str(args.workers)

    # 5) Manifest for paper traceability
    writer.write(
        paths.manifest_path(),
        build_manifest(
            experiment_cfg={
                "experiment": _asdict_safe(exp_cfg),
                "mode": args.mode,
                "config_path": args.config,
            },
            code_paths=["main.py"],
            extra={"system": _sys_metadata()},
        ),
    )

    # 6) Generate signals (IMPORTANT: generate ALL scenarios here)
    T = _extract_T(cfg_dict, default_T=5.0)

    # NOTE: Most of your pipeline assumes DSP-rate waveforms.
    # If you want physics-rate generation + downsample, do it in scenarios or in runners,
    # but keep main deterministic and simple.
    signals = get_test_signals(
        fs=float(exp_cfg.fs_dsp_hz), T=float(T), seed=int(exp_cfg.seed)
    )

    # Accept both dict and list outputs defensively
    if signals is None:
        signals = {}
    n_scen = len(signals) if hasattr(signals, "__len__") else 0

    print(f"🔍 Escenarios detectados: {n_scen}")
    if n_scen == 0:
        print("[ERROR] No se generaron señales. Revisa scenarios/.")
        return 1

    # =========================================================
    # MONTE CARLO MODE (single call; mc.py handles parallelism)
    # =========================================================
    if args.mode == "mc":
        try:
            from experiments import MonteCarloRunner
        except Exception as e:
            print(f"[ERROR] No se pudo importar MonteCarloRunner: {e}")
            return 1

        print(
            "🚀 Ejecutando MonteCarloRunner sobre TODOS los escenarios (paralelismo interno en mc.py)."
        )

        registry = build_registry(registry_cfg)
        mc_runner = MonteCarloRunner(exp_cfg, registry)

        # ✅ Pass ALL scenarios together so plan.json gets the full scenario set
        result = mc_runner.run(signals)

        mc_path = os.path.join(paths.results_mc_root(), "mc_results.json")
        writer.write(mc_path, result)
        print(f"💾 MC guardado en: {mc_path}")
        return 0

    # =========================================================
    # SINGLE MODE (sequential debug)
    # =========================================================
    if args.mode == "single":
        from experiments import BenchmarkRunner

        registry = build_registry(registry_cfg)
        runner = BenchmarkRunner(exp_cfg, registry)
        results = runner.run(signals)

        out_path = os.path.join(paths.figures_root(), "benchmark_results.json")
        writer.write(out_path, results)
        print(f"✅ Single mode complete. Saved: {out_path}")
        return 0

    # =========================================================
    # FIGURES MODE (placeholder)
    # =========================================================
    if args.mode == "figures":
        print(
            "ℹ️ Mode 'figures' no implementado aún en main.py (usa tus scripts de dashboards)."
        )
        return 0

    return 0


if __name__ == "__main__":
    # Important on macOS: spawn requires this block
    raise SystemExit(main())
