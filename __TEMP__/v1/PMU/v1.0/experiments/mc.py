#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc.py — Monte Carlo Benchmark Runner (Q1-grade, stage-based, disk-streaming, optional parallel eval)

Q1 BULLETPROOF UPGRADES IN THIS VERSION (vs your pasted file)
-------------------------------------------------------------
1) **Latencia justa**: exporta métricas RAW + métricas ALIGNED (compensadas por latency_samples).
   - Sufijo: *_ALIGNED
   - Política: shift f_hat by latency_samples and crop to common length.
   - Trip-time se mantiene RAW (porque cuenta desde t=0 real).

2) **Warm-up trazable**: exporta warm_up_samples, eval_samples, eval_duration_s.
   - Warm-up definido como 0.1s (consistente con compute_metrics).

3) **Perturbation stats completos**: exporta stats efectivos (snr/noise, timing shift, gain,
   harmonics amps, impulses_count, clip_frac).

4) **Anti-leak / reproducibilidad**: plan incluye hashes:
   - tuning_dataset_hash (train seeds + oracle/global tune fractions + train scenario set)
   - evaluation_dataset_hash (test seeds + eval scenario set)

5) **Timing protocol opcional (Q1)**: measure_timing_strict (serial) genera:
   artifacts/results_mc/exports/timing_strict.csv

NO CAMBIA contratos públicos de tus runners/estimators.
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import pickle
import traceback
from dataclasses import dataclass, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.config import ExperimentConfig
from experiments.registry import MethodRegistry
from experiments.tuning import build_grids
from experiments.io import save_json

from domain.metrics_api import compute_metrics
from domain.metrics_base import MetricConfig

from scenarios.ibg_events import get_test_signal


# ============================================================
# Logging (LOUD, timestamped)
# ============================================================


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _short(x: Any, n: int = 180) -> str:
    s = str(x)
    return s if len(s) <= n else (s[:n] + "…")


def _dump_obj(name: str, obj: Any, max_attrs: int = 50) -> None:
    try:
        if isinstance(obj, dict):
            _log(
                f"[DUMP] {name}: type=dict(len={len(obj)}) preview=dict keys={list(obj.keys())[:12]}"
            )
        else:
            _log(f"[DUMP] {name}: type={type(obj).__name__} preview={_short(obj)}")

        attrs = []
        try:
            attrs = [a for a in dir(obj) if not a.startswith("_")]
        except Exception:
            attrs = []
        if attrs:
            _log(f"[DUMP] {name}: attrs(head {max_attrs})={attrs[:max_attrs]}")
    except Exception as e:
        _log(f"[DUMP] {name}: failed {type(e).__name__}: {e}")


# ============================================================
# Helpers
# ============================================================


def _cfg_get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stable_hash(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _json_dumps_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _job_id(job: Dict[str, Any]) -> str:
    payload = _json_dumps_compact(job)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def _percentiles(x: np.ndarray, ps: List[int]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def _metric_obj_to_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        try:
            if x.shape == ():
                v = float(x.item())
                return v if np.isfinite(v) else None
            if x.size == 1:
                v = float(np.ravel(x)[0])
                return v if np.isfinite(v) else None
        except Exception:
            return None
    if isinstance(x, dict):
        for k in ("raw", "value", "mean", "val", "score"):
            if k in x:
                return _metric_obj_to_float(x[k])
    for attr in ("raw", "value", "mean", "score"):
        if hasattr(x, attr):
            try:
                return _metric_obj_to_float(getattr(x, attr))
            except Exception:
                pass
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _warmup_samples(fs: float) -> int:
    # Must match metrics_api.compute_metrics warm-up policy (0.1s)
    return int(round(0.1 * float(fs)))


def _align_by_latency(
    f_hat: np.ndarray, f_true: np.ndarray, latency_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Latency-compensation policy for fair accuracy comparisons.
    - Shift f_hat forward by latency_samples (drop first latency samples),
      and crop f_true accordingly to common length.
    """
    fh = np.asarray(f_hat, dtype=float).reshape(-1)
    ft = np.asarray(f_true, dtype=float).reshape(-1)
    L = int(min(fh.size, ft.size))
    if L <= 0:
        return fh[:0], ft[:0]

    fh = fh[:L]
    ft = ft[:L]

    d = int(max(0, latency_samples))
    if d == 0:
        return fh, ft

    if d >= L:
        return fh[:0], ft[:0]

    # shift: fh(t) corresponds to ft(t + d)
    fh_a = fh[d:]
    ft_a = ft[:-d]
    L2 = int(min(fh_a.size, ft_a.size))
    return fh_a[:L2], ft_a[:L2]


def _hash_dataset(obj: Any) -> str:
    s = _json_dumps_compact(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Robust JSON serialization (for cfg/registry snapshots)
# ============================================================


def _safe_jsonify(x: Any, *, max_depth: int = 6, _depth: int = 0) -> Any:
    if _depth >= max_depth:
        return str(type(x).__name__)

    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        if isinstance(x, float) and (not np.isfinite(x)):
            return None
        return x

    if isinstance(x, (np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None

    if isinstance(x, np.ndarray):
        if x.size <= 2000:
            return _safe_jsonify(x.tolist(), max_depth=max_depth, _depth=_depth + 1)
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}

    if isinstance(x, (list, tuple)):
        return [
            _safe_jsonify(v, max_depth=max_depth, _depth=_depth + 1) for v in x[:5000]
        ]

    if isinstance(x, dict):
        out = {}
        for k, v in list(x.items())[:10000]:
            kk = str(k)
            out[kk] = _safe_jsonify(v, max_depth=max_depth, _depth=_depth + 1)
        return out

    if is_dataclass(x):
        try:
            return _safe_jsonify(asdict(x), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            return str(x)

    if hasattr(x, "model_dump"):
        try:
            return _safe_jsonify(x.model_dump(), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return _safe_jsonify(x.dict(), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            pass

    if hasattr(x, "__dict__"):
        try:
            return _safe_jsonify(vars(x), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            return str(x)

    try:
        return str(x)
    except Exception:
        return str(type(x).__name__)


# ============================================================
# Robust GridConfig / object conversion + extraction
# ============================================================


def _is_nonempty_dict(x: Any) -> bool:
    return isinstance(x, dict) and len(x) > 0


def _try_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _try_read_yaml(path: str) -> Optional[dict]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        _log(f"[GRIDS][LOAD] YAML requested but PyYAML not installed. error={e}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _object_to_mapping(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        p = os.path.expanduser(obj.strip())
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.exists(p):
            return {}
        ext = os.path.splitext(p)[1].lower()
        if ext == ".json":
            j = _try_read_json(p)
            return j if isinstance(j, dict) else {}
        if ext in (".yaml", ".yml"):
            y = _try_read_yaml(p)
            return y if isinstance(y, dict) else {}
        return {}

    if is_dataclass(obj):
        try:
            d = asdict(obj)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    if hasattr(obj, "model_dump"):
        try:
            d = obj.model_dump()
            return d if isinstance(d, dict) else {}
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            d = obj.dict()
            return d if isinstance(d, dict) else {}
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            return {}

    return {}


def _extract_grids_dict(d: dict) -> dict:
    if not isinstance(d, dict) or len(d) == 0:
        return {}

    wrapper_keys = [
        "grids",
        "methods",
        "tuners",
        "tuner_grids",
        "per_method",
        "grid",
        "grid_map",
    ]
    for k in wrapper_keys:
        if k in d and isinstance(d[k], dict) and len(d[k]) > 0:
            return d[k]

    for k in ("path", "file", "filepath", "json_path", "yaml_path"):
        if k in d and isinstance(d[k], str) and d[k].strip():
            p = os.path.expanduser(d[k].strip())
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            if os.path.exists(p) and os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext == ".json":
                    j = _try_read_json(p)
                    if isinstance(j, dict):
                        return _extract_grids_dict(j)
                if ext in (".yaml", ".yml"):
                    y = _try_read_yaml(p)
                    if isinstance(y, dict):
                        return _extract_grids_dict(y)

    for _, v in d.items():
        if isinstance(v, dict) and any(kk in v for kk in wrapper_keys):
            inner = _extract_grids_dict(v)
            if isinstance(inner, dict) and len(inner) > 0:
                return inner

    return d


def _load_grid_source(grids_src: Any, name: str = "cfg.grids") -> Dict[str, Any]:
    base = _object_to_mapping(grids_src)
    extracted = _extract_grids_dict(base)

    _dump_obj(name, grids_src)
    if is_dataclass(grids_src):
        try:
            _log(
                f"[DUMP] {name}: dataclass asdict keys(head 60)={list(asdict(grids_src).keys())[:60]}"
            )
        except Exception:
            pass
    if hasattr(grids_src, "__dict__"):
        try:
            _log(
                f"[DUMP] {name}: __dict__ keys(head 80)={list(vars(grids_src).keys())[:80]}"
            )
        except Exception:
            pass

    if isinstance(base, dict):
        _log(
            f"[DUMP] {name}.base_mapping: type=dict(len={len(base)}) preview=dict keys={list(base.keys())[:8]}"
        )
        _log(
            f"[DUMP] {name}.base_mapping: keys_count={len(base)} keys_head={list(base.keys())[:40]}"
        )
        for kk in list(base.keys())[:8]:
            vv = base[kk]
            if isinstance(vv, (list, tuple, np.ndarray)):
                _log(
                    f"[DUMP] {name}.base_mapping[{kk!r}]: type={type(vv).__name__}(len={len(vv)}) preview=list head={list(vv)[:8]}"
                )
    if isinstance(extracted, dict):
        _log(
            f"[DUMP] {name}.extracted: type=dict(len={len(extracted)}) preview=dict keys={list(extracted.keys())[:8]}"
        )
        _log(
            f"[DUMP] {name}.extracted: keys_count={len(extracted)} keys_head={list(extracted.keys())[:40]}"
        )
        for kk in list(extracted.keys())[:8]:
            vv = extracted[kk]
            if isinstance(vv, (list, tuple, np.ndarray)):
                _log(
                    f"[DUMP] {name}.extracted[{kk!r}]: type={type(vv).__name__}(len={len(vv)}) preview=list head={list(vv)[:8]}"
                )
    else:
        _log(f"[DUMP] {name}.extracted: non-dict type={type(extracted).__name__}")

    if not _is_nonempty_dict(extracted) and grids_src is not None:
        _log(
            f"[GRIDS][HINT] {name}: extracted is empty. Likely wrong wrapper key OR object has fields not captured."
        )
        try:
            hint = _object_to_mapping(grids_src)
            _log(f"[GRIDS][HINT] converted keys={list(hint.keys())[:30]}")
        except Exception as e:
            _log(f"[GRIDS][HINT] failed: {type(e).__name__}: {e}")

    return extracted if isinstance(extracted, dict) else {}


# ============================================================
# Method-name canonicalization + alias mapping
# ============================================================


def _canon_method_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s


def _build_method_alias_map(
    methods_cfg: List[str], grids: Dict[str, Any]
) -> Dict[str, str]:
    if not isinstance(grids, dict):
        return {m: m for m in methods_cfg}

    by_canon: Dict[str, List[str]] = {}
    for k in grids.keys():
        by_canon.setdefault(_canon_method_name(k), []).append(k)

    out: Dict[str, str] = {}
    for m in methods_cfg:
        if m in grids:
            out[m] = m
            continue
        cm = _canon_method_name(m)
        cands = by_canon.get(cm, [])
        if not cands:
            out[m] = m
            continue
        if len(cands) == 1:
            out[m] = cands[0]
            continue
        out[m] = sorted(cands, key=lambda x: (len(x), x))[0]
    return out


# ============================================================
# Perturb key normalization (supports your aliases across project)
# ============================================================


def _normalize_perturb_keys(pert: Any) -> Dict[str, Any]:
    if pert is None:
        return {}
    if not isinstance(pert, dict):
        try:
            pert = {k: getattr(pert, k) for k in dir(pert) if not k.startswith("_")}
        except Exception:
            return {}
    p = dict(pert)

    alias = {
        "amp_jitter_pct": "amp_pct_jitter",
        "amp_jitter": "amp_pct_jitter",
        "snr_jitter_db": "snr_db_jitter",
        "noise_pct": "noise_rms_pct",
        "noise_cap_pct": "noise_rms_cap_pct",
        "impulse_prob": "impulsive_prob",
        "impulse_scale": "impulsive_scale",
        "impulse_scale_pct": "impulsive_scale_pct",
        "impulse_cap_pct": "impulsive_cap_pct",
        "harmonics_jitter_pct": "harmonics_pct_jitter",
        "timing_jitter_samples": "timing_samples_jitter",
        "clip_rms_mult": "hard_clip_rms_mult",
    }
    for a, canon in alias.items():
        if canon not in p and a in p:
            p[canon] = p[a]
    return p


# ============================================================
# Tuning theta sanitizer (FAIL-FAST, LOUD)
# ============================================================


def _sanitize_theta_strict(
    theta: Dict[str, Any], *, method: str, stage: str, scenario: Optional[str]
) -> Dict[str, Any]:
    if not theta:
        return {}

    out: Dict[str, Any] = {}
    for k, v in theta.items():
        if k == "label":
            continue

        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError(
                    f"[TUNING GRID ERROR] empty list for param '{k}' "
                    f"(method={method}, stage={stage}, scenario={scenario}). theta={theta}"
                )
            if len(v) == 1:
                out[k] = v[0]
            else:
                raise ValueError(
                    f"[TUNING GRID ERROR] list-valued param '{k}' still present (len={len(v)}). "
                    f"method={method}, stage={stage}, scenario={scenario}\n"
                    f"offending theta={theta}"
                )
            continue

        if isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError(
                    f"[TUNING GRID ERROR] empty ndarray for param '{k}' "
                    f"(method={method}, stage={stage}, scenario={scenario}). theta={theta}"
                )
            if v.size == 1:
                out[k] = float(np.ravel(v)[0])
            else:
                raise ValueError(
                    f"[TUNING GRID ERROR] ndarray-valued param '{k}' still present (size={v.size}). "
                    f"method={method}, stage={stage}, scenario={scenario}\n"
                    f"offending theta={theta}"
                )
            continue

        out[k] = v

    return out


# ============================================================
# Streaming writer (JSONL)
# ============================================================


class JsonlWriter:
    def __init__(self, path: str, flush_every: int = 200) -> None:
        self.path = path
        _ensure_dir(os.path.dirname(path))
        self.f = open(path, "a", encoding="utf-8")
        self.flush_every = int(flush_every)
        self._n = 0

    def append(self, record: Dict[str, Any]) -> None:
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()


# ============================================================
# PARAM-GRID -> METHOD-GRID auto wrapper (fix for your crash)
# ============================================================


def _is_param_grid(d: Any, methods: List[str]) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    mset = set(map(str, methods))
    keys = list(d.keys())
    if any(str(k) in mset for k in keys):
        return False
    list_vals = 0
    for v in d.values():
        if isinstance(v, (list, tuple, np.ndarray)):
            list_vals += 1
    return list_vals >= max(1, int(0.6 * len(d)))


def _cartesian_product(param_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    items = [(k, list(v)) for k, v in param_lists.items() if v is not None]
    if not items:
        return [{}]
    items = [(k, v) for (k, v) in items if isinstance(v, list) and len(v) > 0]
    if not items:
        return [{}]

    grids: List[Dict[str, Any]] = [{}]
    for k, vals in items:
        new_grids: List[Dict[str, Any]] = []
        for g in grids:
            for x in vals:
                gg = dict(g)
                gg[k] = x
                new_grids.append(gg)
        grids = new_grids
    return grids


def _det_subsample_grid(
    grid: List[Dict[str, Any]], max_n: int, seed: int
) -> List[Dict[str, Any]]:
    if not isinstance(grid, list):
        return []
    if max_n <= 0 or len(grid) <= max_n:
        return grid
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    idx = rng.choice(len(grid), size=int(max_n), replace=False)
    idx = np.sort(idx)
    return [grid[int(i)] for i in idx]


def _wrap_param_grid_to_method_grid(
    param_grid: Dict[str, Any],
    methods: List[str],
    *,
    cap_per_method: int = 5000,
    cap_seed: int = 777,
) -> Dict[str, List[Dict[str, Any]]]:
    pg: Dict[str, List[Any]] = {}
    for k, v in param_grid.items():
        if isinstance(v, np.ndarray):
            v = list(np.ravel(v).tolist())
        if isinstance(v, (list, tuple)):
            pg[str(k)] = list(v)
        else:
            pg[str(k)] = [v]

    KF_METHODS = {"RA-EKF2", "RA-EKF", "CKF", "UKF", "IEKF", "EnKF", "EKF", "LKF"}
    PLL_METHODS = {"SRF-PLL", "MAF-SRF-PLL", "DDSRF-PLL"}
    SOGI_METHODS = {"MSOGI-FLL", "SOGI-Industrial", "SOGI-Classic"}

    method_params: Dict[str, List[str]] = {}
    for m in methods:
        if m in KF_METHODS:
            method_params[m] = ["kf_q", "kf_r"]
        elif m in PLL_METHODS:
            method_params[m] = ["pll_kp", "pll_ki"]
        elif m in SOGI_METHODS:
            method_params[m] = ["sogi_k", "sogi_g"]
        elif m == "IpDFT":
            method_params[m] = ["ipdft_cycles", "ipdft_decim", "ipdft_window_type"]
        elif m == "TFT":
            method_params[m] = ["tft_win"]
        elif m == "RLS":
            method_params[m] = ["rls_lam", "rls_win"]
        elif m == "RLS-VFF":
            method_params[m] = ["vff_lam_min", "vff_ka", "vff_win_smooth", "vff_decim"]
        elif m == "Teager":
            method_params[m] = ["teager_win"]
        else:
            method_params[m] = []

    out: Dict[str, List[Dict[str, Any]]] = {}
    for m in methods:
        keys = method_params.get(m, [])
        plist = {k: pg[k] for k in keys if k in pg}
        grid = _cartesian_product(plist)
        grid = _det_subsample_grid(
            grid, max_n=int(cap_per_method), seed=_stable_hash(f"{cap_seed}|{m}")
        )
        out[m] = grid

    return out


def _grid_kind(d: Any, methods: List[str]) -> str:
    if not isinstance(d, dict):
        return f"non-dict({type(d).__name__})"
    if _is_param_grid(d, methods):
        hits = sum(
            1
            for k in d.keys()
            if str(k)
            in {
                "kf_q",
                "kf_r",
                "pll_kp",
                "pll_ki",
                "sogi_k",
                "sogi_g",
                "rls_lam",
                "rls_win",
                "ipdft_cycles",
            }
        )
        return f"param-grid(param_hits={hits})"
    mset = set(map(str, methods))
    if any(str(k) in mset for k in d.keys()):
        return "method-grid"
    return "unknown-dict"


# ============================================================
# Job schema
# ============================================================


@dataclass(frozen=True)
class RunJob:
    stage: str
    scenario: str
    seed: int
    seed_role: str
    method: str
    mode: str
    theta: Dict[str, Any]
    delta_pct: float = 0.0
    perturb_id: Optional[int] = None
    export_waveforms: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "stage": self.stage,
            "scenario": self.scenario,
            "seed": int(self.seed),
            "seed_role": str(self.seed_role),
            "method": self.method,
            "mode": self.mode,
            "theta": self.theta,
            "delta_pct": float(self.delta_pct),
            "perturb_id": None if self.perturb_id is None else int(self.perturb_id),
            "export_waveforms": bool(self.export_waveforms),
        }
        d["job_id"] = _job_id(d)
        return d


# ============================================================
# Worker (optional multiprocessing)
# ============================================================


def _worker_run_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg: ExperimentConfig = pickle.loads(payload["cfg_pkl"])
    registry: MethodRegistry = pickle.loads(payload["registry_pkl"])
    metric_cfg: MetricConfig = pickle.loads(payload["metric_cfg_pkl"])
    job_d: Dict[str, Any] = payload["job"]
    mc_run_id: str = str(payload["mc_run_id"])

    job = RunJob(
        stage=job_d["stage"],
        scenario=job_d["scenario"],
        seed=int(job_d["seed"]),
        seed_role=str(job_d.get("seed_role", "test")),
        method=job_d["method"],
        mode=job_d["mode"],
        theta=dict(job_d.get("theta", {})),
        delta_pct=float(job_d.get("delta_pct", 0.0)),
        perturb_id=job_d.get("perturb_id", None),
        export_waveforms=bool(job_d.get("export_waveforms", False)),
    )

    def _scenario_T_local() -> float:
        sc = getattr(cfg, "scenario", None)
        if isinstance(sc, dict) and "T" in sc:
            return float(sc["T"])
        return 5.0

    def _regen_scenario_local(scenario_id: str, seed: int):
        fs_phys = float(getattr(cfg, "fs_physics_hz", cfg.fs_dsp_hz))
        T = float(_scenario_T_local())
        return get_test_signal(scenario_id=scenario_id, fs=fs_phys, T=T, seed=int(seed))

    def _perturb_voltage_local(
        v: np.ndarray, rng: np.random.Generator, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        sc_id = str(meta.get("scenario_id", ""))
        mc = getattr(cfg, "mc", None)
        if not mc:
            return np.asarray(v, dtype=float).copy(), {
                "clip_frac": 0.0,
                "timing_shift_samples": 0.0,
            }

        pert_raw = _cfg_get(mc, "perturb", None)
        pert = _normalize_perturb_keys(pert_raw)
        if not pert:
            return np.asarray(v, dtype=float).copy(), {
                "clip_frac": 0.0,
                "timing_shift_samples": 0.0,
            }

        out = np.asarray(v, dtype=float).copy()
        n = out.size
        if n == 0:
            return out, {"clip_frac": 0.0, "timing_shift_samples": 0.0}

        stats: Dict[str, float] = {
            "clip_frac": 0.0,
            "timing_shift_samples": 0.0,
            "amp_gain": 1.0,
            "noise_std": 0.0,
            "snr_db_effective": float("nan"),
            "harm_a3": 0.0,
            "harm_a5": 0.0,
            "harm_a7": 0.0,
            "impulses_count": 0.0,
        }

        sig_rms = float(np.sqrt(np.mean(out * out) + 1e-12))

        # Timing shift
        K_shift = int(_cfg_get(pert, "timing_samples_jitter", 0))
        if K_shift > 0:
            s = int(rng.integers(-K_shift, K_shift + 1))
            if s != 0:
                out = np.roll(out, s)
            stats["timing_shift_samples"] = float(s)

        # Gain jitter
        amp_std = float(_cfg_get(pert, "amp_pct_jitter", 0.0))
        amp_cap = float(_cfg_get(pert, "amp_cap_pct", 0.05))
        if amp_std > 0:
            g = float(1.0 + rng.normal(0.0, amp_std))
            g = float(np.clip(g, 1.0 - amp_cap, 1.0 + amp_cap))
            out *= g
            stats["amp_gain"] = float(g)

        # Noise (either noise_rms_pct or SNR)
        add_awgn = "Pure" not in sc_id
        if add_awgn:
            noise_rms_pct = _cfg_get(pert, "noise_rms_pct", None)
            if noise_rms_pct is not None:
                noise_rms_pct = float(noise_rms_pct)
                noise_rms_cap = float(_cfg_get(pert, "noise_rms_cap_pct", 0.03))
                nr = float(np.clip(noise_rms_pct, 0.0, noise_rms_cap))
                noise_std = nr * sig_rms
                if noise_std > 0:
                    out += rng.normal(0.0, noise_std, size=out.shape)
                stats["noise_std"] = float(noise_std)
                stats["snr_db_effective"] = float("nan")
            else:
                snr_jit = _cfg_get(pert, "snr_db_jitter", None)
                if snr_jit is not None and float(snr_jit) > 0:
                    snr_base = float(_cfg_get(pert, "snr_db", 40.0))
                    snr_db = float(snr_base + rng.normal(0.0, float(snr_jit)))
                    snr_db = float(np.clip(snr_db, 10.0, 80.0))
                    noise_std = sig_rms / (10.0 ** (snr_db / 20.0))
                    noise_std = float(np.clip(noise_std, 0.0, 0.03 * sig_rms))
                    if noise_std > 0:
                        out += rng.normal(0.0, noise_std, size=out.shape)
                    stats["noise_std"] = float(noise_std)
                    stats["snr_db_effective"] = float(snr_db)

        # Harmonics
        harm_jit = float(_cfg_get(pert, "harmonics_pct_jitter", 0.0))
        if harm_jit > 0:
            fs = float(getattr(cfg, "fs_dsp_hz", 1.0))
            f0 = float(_cfg_get(pert, "nominal_f_hz", 60.0))
            t = np.arange(n, dtype=float) / max(fs, 1.0)
            a3 = float(np.clip(abs(rng.normal(0.0, harm_jit)), 0.0, 0.10))
            a5 = float(np.clip(abs(rng.normal(0.0, harm_jit * 0.7)), 0.0, 0.07))
            a7 = float(np.clip(abs(rng.normal(0.0, harm_jit * 0.5)), 0.0, 0.05))
            p3 = float(rng.uniform(0.0, 2.0 * np.pi))
            p5 = float(rng.uniform(0.0, 2.0 * np.pi))
            p7 = float(rng.uniform(0.0, 2.0 * np.pi))
            harm = (
                (a3 * sig_rms) * np.sin(2.0 * np.pi * (3.0 * f0) * t + p3)
                + (a5 * sig_rms) * np.sin(2.0 * np.pi * (5.0 * f0) * t + p5)
                + (a7 * sig_rms) * np.sin(2.0 * np.pi * (7.0 * f0) * t + p7)
            )
            out = out + harm
            stats["harm_a3"] = float(a3)
            stats["harm_a5"] = float(a5)
            stats["harm_a7"] = float(a7)

        # Impulses
        allow_impulses_in = _cfg_get(pert, "allow_impulses_in", None)
        if not isinstance(allow_impulses_in, (list, tuple)):
            allow_impulses_in = ["G3_E13_Impulsive_Outliers"]
        if sc_id in set(map(str, allow_impulses_in)):
            p_imp = float(_cfg_get(pert, "impulsive_prob", 0.0))
            if p_imp > 0:
                scale_pct = _cfg_get(pert, "impulsive_scale_pct", None)
                if scale_pct is not None:
                    scale_pct = float(scale_pct)
                else:
                    imp_scale = _cfg_get(pert, "impulsive_scale", 1.0)
                    scale_pct = float(imp_scale) / 100.0
                scale_pct = float(np.clip(scale_pct, 0.0, 0.20))
                cap_pct = float(_cfg_get(pert, "impulsive_cap_pct", 0.05))
                cap_pct = float(np.clip(cap_pct, 0.0, 0.50))
                mask = rng.random(size=out.shape) < p_imp
                k = int(np.sum(mask))
                if k > 0:
                    spikes = rng.normal(0.0, scale_pct * sig_rms, size=k)
                    lim2 = cap_pct * sig_rms
                    spikes = np.clip(spikes, -lim2, +lim2)
                    out[mask] += spikes
                stats["impulses_count"] = float(k)

        # Hard clip
        hard_cap = float(_cfg_get(pert, "hard_clip_rms_mult", 6.0))
        lim = hard_cap * sig_rms
        before = out.copy()
        out = np.clip(out, -lim, +lim)
        clip_frac = float(np.mean(before != out)) if out.size > 0 else 0.0
        stats["clip_frac"] = float(clip_frac)

        return out, stats

    seed = int(job.seed)
    rng = np.random.default_rng(seed)

    try:
        t_phys, v_ana, f_true, meta = _regen_scenario_local(job.scenario, seed=seed)
        ratio = int(cfg.downsampling_ratio)
        v_ds, f_ds, t_ds = v_ana[::ratio], f_true[::ratio], t_phys[::ratio]
        v_eval, pert_stats = _perturb_voltage_local(v_ds, rng, meta)
        f_eval = f_ds
    except Exception as e:
        pair_id = _job_id(
            {
                "stage": job.stage,
                "scenario": job.scenario,
                "seed": seed,
                "seed_role": job.seed_role,
                "mode": job.mode,
                "delta_pct": float(job.delta_pct),
                "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
            }
        )
        return {
            "mc_run_id": mc_run_id,
            "pair_id": pair_id,
            "job_id": _job_id(job.to_dict()),
            "stage": job.stage,
            "scenario": job.scenario,
            "scenario_id": job.scenario,
            "seed": seed,
            "seed_role": job.seed_role,
            "method": job.method,
            "method_family": "unknown",
            "mode": job.mode,
            "delta_pct": float(job.delta_pct),
            "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
            "fs_dsp_hz": float(getattr(cfg, "fs_dsp_hz", 0.0)),
            "downsampling_ratio": int(getattr(cfg, "downsampling_ratio", 1)),
            "latency_samples": 0,
            "exec_time_s": float("nan"),
            "clipped_frac": 0.0,
            "nan_frac": 1.0,
            "valid_trace": False,
            "hard_fail": True,
            "theta": job.theta,
            "_error": f"scenario_gen_failed: {type(e).__name__}: {e}",
        }

    spec = registry.get(job.method)
    make_est = spec.builder
    base_params = {"fs_hz": float(cfg.fs_dsp_hz)}
    method_family = str(getattr(spec, "family", "unknown"))

    t_start = time.perf_counter()
    err = None
    try:
        est = make_est({**base_params, **(job.theta or {})})
        if hasattr(est, "reset"):
            est.reset()
        f_hat = np.array([est.step(float(x)) for x in v_eval], dtype=float)
    except Exception as e:
        err = e
        f_hat = np.full_like(f_eval, np.nan, dtype=float)
    exec_t = float(time.perf_counter() - t_start)

    latency = 0
    try:
        if getattr(spec, "structural_latency", None) is not None:
            latency = int(spec.structural_latency(est))  # type: ignore[name-defined]
    except Exception:
        latency = 0

    # RAW metrics (as-is)
    try:
        mets_raw = compute_metrics(
            f_hat, f_eval, exec_t, latency, metric_cfg, job.scenario
        )
    except Exception as e:
        mets_raw = {"_metrics_error": f"{type(e).__name__}: {e}"}

    # ALIGNED metrics (latency-compensated)
    try:
        f_hat_a, f_eval_a = _align_by_latency(f_hat, f_eval, latency)
        mets_al = compute_metrics(
            f_hat_a, f_eval_a, exec_t, 0, metric_cfg, job.scenario
        )
    except Exception as e:
        mets_al = {"_metrics_error": f"{type(e).__name__}: {e}"}

    nan_frac = float(np.mean(~np.isfinite(f_hat))) if f_hat.size > 0 else 1.0
    valid_trace = bool(
        np.all(np.isfinite(f_hat)) and f_hat.size == f_eval.size and f_hat.size > 0
    )
    hard_fail = bool((not valid_trace) or (err is not None))

    scenario_id = str(meta.get("scenario_id", job.scenario))
    pair_id = _job_id(
        {
            "stage": job.stage,
            "scenario": job.scenario,
            "seed": seed,
            "seed_role": job.seed_role,
            "mode": job.mode,
            "delta_pct": float(job.delta_pct),
            "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
        }
    )

    fs_dsp = float(getattr(cfg, "fs_dsp_hz", 0.0))
    wup = _warmup_samples(fs_dsp)
    Ltot = int(min(np.asarray(f_hat).size, np.asarray(f_eval).size))
    wup_eff = int(wup if Ltot > wup else 0)
    eval_samples = int(max(0, Ltot - wup_eff))

    rec: Dict[str, Any] = {
        "mc_run_id": mc_run_id,
        "pair_id": pair_id,
        "job_id": _job_id(job.to_dict()),
        "stage": job.stage,
        "scenario": job.scenario,
        "scenario_id": scenario_id,
        "seed": seed,
        "seed_role": job.seed_role,
        "method": job.method,
        "method_family": method_family,
        "mode": job.mode,
        "delta_pct": float(job.delta_pct),
        "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
        "fs_dsp_hz": fs_dsp,
        "downsampling_ratio": int(getattr(cfg, "downsampling_ratio", 1)),
        "latency_samples": int(latency),
        "effective_latency_s": float(latency) / max(fs_dsp, 1.0),
        "alignment_policy": "shift_fhat_by_latency_then_crop",
        "warm_up_samples": int(wup_eff),
        "eval_samples": int(eval_samples),
        "eval_duration_s": float(eval_samples) / max(fs_dsp, 1.0),
        # perturbation stats
        "clipped_frac": float(pert_stats.get("clip_frac", 0.0)),
        "pert_timing_shift_samples": float(pert_stats.get("timing_shift_samples", 0.0)),
        "pert_amp_gain": float(pert_stats.get("amp_gain", 1.0)),
        "pert_noise_std": float(pert_stats.get("noise_std", 0.0)),
        "pert_snr_db_effective": float(
            pert_stats.get("snr_db_effective", float("nan"))
        ),
        "pert_harm_a3": float(pert_stats.get("harm_a3", 0.0)),
        "pert_harm_a5": float(pert_stats.get("harm_a5", 0.0)),
        "pert_harm_a7": float(pert_stats.get("harm_a7", 0.0)),
        "pert_impulses_count": float(pert_stats.get("impulses_count", 0.0)),
        "exec_time_s": float(exec_t),
        "nan_frac": float(nan_frac),
        "valid_trace": bool(valid_trace),
        "hard_fail": bool(hard_fail),
        "theta": job.theta,
    }
    if err is not None:
        rec["_error"] = f"estimator_failed: {type(err).__name__}: {err}"

    # flatten RAW metrics
    for k, v in mets_raw.items():
        num = _metric_obj_to_float(v)
        if num is not None:
            rec[k] = float(num)

    # flatten ALIGNED metrics with suffix (avoid duplicating trip-time + compute-cost)
    # Keep aligned versions for accuracy/dynamics/tails only.
    skip_prefixes = ("TRIP_TIME_", "TIME_PER_SAMPLE_US", "LATENCY_SAMPLES")
    for k, v in mets_al.items():
        if any(str(k).startswith(p) for p in skip_prefixes):
            continue
        if str(k).startswith("_"):
            continue
        num = _metric_obj_to_float(v)
        if num is not None:
            rec[f"{k}_ALIGNED"] = float(num)

    return rec


# ============================================================
# Runner
# ============================================================


class MonteCarloRunner:
    def __init__(self, cfg: ExperimentConfig, registry: MethodRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.metric_cfg = MetricConfig(fs_hz=float(self.cfg.fs_dsp_hz))

        _log(
            "[MC][BOOT] =============================================================="
        )
        _log(
            f"[MC][BOOT] cfg type={type(cfg).__name__} methods_count={len(getattr(cfg,'methods',[]) or [])}"
        )
        _log(f"[MC][BOOT] methods(head)={(getattr(cfg,'methods',[]) or [])[:25]}")
        _log(
            "[MC][BOOT] =============================================================="
        )

        tuners_obj = getattr(cfg, "tuners", None)
        grids_obj = getattr(cfg, "grids", None)

        _log("[GRIDS][LOAD] ---- loading cfg.tuners ----")
        tuners_map = _load_grid_source(tuners_obj, name="cfg.tuners")
        _log("[GRIDS][LOAD] ---- done cfg.tuners ----")

        _log("[GRIDS][LOAD] ---- loading cfg.grids ----")
        grids_map = _load_grid_source(grids_obj, name="cfg.grids")
        _log("[GRIDS][LOAD] ---- done cfg.grids ----")

        methods_cfg = list(getattr(self.cfg, "methods", []))
        mc = getattr(self.cfg, "mc", {}) or {}

        use_tuners = _is_nonempty_dict(tuners_map)
        source_name = "cfg.tuners" if use_tuners else "cfg.grids"
        grids_src = tuners_map if use_tuners else grids_map

        _log("[GRIDS] ==============================================================")
        _log(f"[GRIDS] selected source={source_name}")
        _log(
            f"[GRIDS] tuners_map keys={len(tuners_map)} kind={_grid_kind(tuners_map, methods_cfg)}"
        )
        _log(
            f"[GRIDS] grids_map  keys={len(grids_map)} kind={_grid_kind(grids_map, methods_cfg)}"
        )
        _log(
            f"[GRIDS] selected grids_src keys={len(grids_src) if isinstance(grids_src, dict) else 'n/a'} kind={_grid_kind(grids_src, methods_cfg)}"
        )
        if isinstance(grids_src, dict):
            _log(f"[GRIDS] grids_src keys_head={list(grids_src.keys())[:30]}")
        _log("[GRIDS] ==============================================================")

        cap_per_method = int(_cfg_get(mc, "grid_cap_per_method", 5000))
        cap_seed = int(_cfg_get(mc, "base_seed", 777))
        kind = _grid_kind(grids_src, methods_cfg)
        _log(f"[GRIDS] detected kind={kind}")

        if isinstance(grids_src, dict) and _is_param_grid(grids_src, methods_cfg):
            _log(
                "[GRIDS] auto-wrapping PARAM-GRID -> METHOD-GRID (fixes ceiling benchmark)."
            )
            self.grids = _wrap_param_grid_to_method_grid(
                grids_src,
                methods_cfg,
                cap_per_method=cap_per_method,
                cap_seed=cap_seed,
            )
        else:
            try:
                self.grids = build_grids(grids_src or {})
            except Exception as e:
                _log(f"[GRIDS] build_grids() crashed: {type(e).__name__}: {e}")
                raise

        _log("[GRIDS] ==============================================================")
        _log(
            f"[GRIDS] final self.grids kind={_grid_kind(self.grids, methods_cfg)} keys={len(self.grids) if isinstance(self.grids, dict) else 'n/a'}"
        )
        if isinstance(self.grids, dict):
            for m in methods_cfg[:25]:
                g = self.grids.get(m, None)
                gl = len(g) if isinstance(g, list) else -1
                _log(f"[GRIDS] method-grid[{m}] len={gl}")
        _log("[GRIDS] ==============================================================")

        self._grid_key_for_method: Dict[str, str] = {}

        self.out_dir = "artifacts/results_mc"
        self.plan_dir = os.path.join(self.out_dir, "plan")
        self.tuning_dir = os.path.join(self.out_dir, "tuning")
        self.runs_dir = os.path.join(self.out_dir, "runs")
        self.exports_dir = os.path.join(self.out_dir, "exports")

        _ensure_dir(self.out_dir)
        _ensure_dir(self.plan_dir)
        _ensure_dir(self.tuning_dir)
        _ensure_dir(self.runs_dir)
        _ensure_dir(self.exports_dir)

        if bool(_cfg_get(mc, "export_grids_snapshot", True)):
            self._export_grids_snapshot(methods_cfg)
        if bool(_cfg_get(mc, "export_methods_catalog", True)):
            self._export_methods_catalog(methods_cfg)
        if bool(_cfg_get(mc, "export_cfg_snapshot", True)):
            self._export_cfg_snapshot()

    def _grid_key(self, method_cfg_name: str) -> str:
        return self._grid_key_for_method.get(method_cfg_name, method_cfg_name)

    def _grid(self, method_cfg_name: str) -> List[Dict[str, Any]]:
        key = self._grid_key(method_cfg_name)
        g = self.grids.get(key, [])
        return g if isinstance(g, list) else []

    # ----------------------------
    # EXPORTS (Q1-proof)
    # ----------------------------

    def _export_cfg_snapshot(self) -> None:
        path = os.path.join(self.exports_dir, "cfg_snapshot.json")
        payload = {
            "cfg_type": type(self.cfg).__name__,
            "cfg": _safe_jsonify(self.cfg),
        }
        save_json(path, payload)
        _log(f"[EXPORT] cfg_snapshot -> {path}")

    def _export_grids_snapshot(self, methods: List[str]) -> None:
        path = os.path.join(self.exports_dir, "grids_snapshot.json")
        if not isinstance(self.grids, dict):
            payload = {
                "grids_type": type(self.grids).__name__,
                "grids": str(self.grids),
            }
            save_json(path, payload)
            _log(f"[EXPORT] grids_snapshot(non-dict) -> {path}")
            return

        snap = {}
        for m in methods:
            g = self.grids.get(m, [])
            if isinstance(g, list):
                exemplars = g[: min(50, len(g))]
                snap[m] = {"len": int(len(g)), "exemplars": _safe_jsonify(exemplars)}
            else:
                snap[m] = {"len": 0, "exemplars": []}

        payload = {
            "grid_kind": _grid_kind(self.grids, methods),
            "methods": list(methods),
            "grids": snap,
        }
        save_json(path, payload)
        _log(f"[EXPORT] grids_snapshot -> {path}")

    def _export_methods_catalog(self, methods: List[str]) -> None:
        rows = []
        catalog = {}

        for m in methods:
            try:
                spec = self.registry.get(m)
            except Exception as e:
                spec = None
                _log(
                    f"[EXPORT][METHODS] registry.get({m}) failed: {type(e).__name__}: {e}"
                )

            fam = (
                str(getattr(spec, "family", "unknown"))
                if spec is not None
                else "unknown"
            )
            has_latency = (
                bool(getattr(spec, "structural_latency", None) is not None)
                if spec is not None
                else False
            )
            builder = getattr(spec, "builder", None) if spec is not None else None
            builder_name = (
                getattr(builder, "__name__", str(type(builder).__name__))
                if builder is not None
                else "None"
            )

            grid_len = (
                len(self.grids.get(m, []))
                if isinstance(self.grids, dict)
                and isinstance(self.grids.get(m, []), list)
                else 0
            )

            entry = {
                "method": m,
                "family": fam,
                "builder": builder_name,
                "has_structural_latency": has_latency,
                "grid_len": int(grid_len),
            }

            param_hints = {}
            try:
                for attr in ("params", "param_names", "hyperparams", "defaults"):
                    if spec is not None and hasattr(spec, attr):
                        param_hints[attr] = _safe_jsonify(getattr(spec, attr))
            except Exception:
                pass
            entry["param_hints"] = param_hints

            catalog[m] = entry
            rows.append(
                {
                    "method": m,
                    "family": fam,
                    "builder": builder_name,
                    "has_structural_latency": int(has_latency),
                    "grid_len": int(grid_len),
                }
            )

        path_json = os.path.join(self.exports_dir, "methods_catalog.json")
        save_json(path_json, {"methods": catalog})
        _log(f"[EXPORT] methods_catalog -> {path_json}")

        try:
            df = pd.DataFrame(rows).sort_values(["family", "method"])
            path_csv = os.path.join(self.exports_dir, "methods_catalog.csv")
            df.to_csv(path_csv, index=False)
            _log(f"[EXPORT] methods_catalog.csv -> {path_csv}")
        except Exception as e:
            _log(f"[EXPORT] methods_catalog.csv failed: {type(e).__name__}: {e}")

    # ----------------------------
    # Hard requirements for "ceiling" benchmarking
    # ----------------------------

    def _assert_grids_for_all_methods(
        self,
        methods: List[str],
        *,
        enable_oracle: bool,
        enable_deployable: bool,
        mc: Dict[str, Any],
    ) -> Tuple[bool, bool]:
        _log("[ASSERT] ==============================================================")
        _log(
            f"[ASSERT] entering _assert_grids_for_all_methods enable_oracle={enable_oracle} enable_deployable={enable_deployable}"
        )
        _log(f"[ASSERT] methods_count={len(methods)} methods(head)={methods[:25]}")
        _log(
            f"[ASSERT] self.grids type={type(self.grids).__name__}(len={len(self.grids) if isinstance(self.grids, dict) else -1})"
        )
        if isinstance(self.grids, dict):
            _log(f"[ASSERT] self.grids keys_head={list(self.grids.keys())[:30]}")
        _log(f"[ASSERT] grid_kind={_grid_kind(self.grids, methods)}")
        _log("[ASSERT] ==============================================================")

        if not (enable_oracle or enable_deployable):
            return enable_oracle, enable_deployable

        if not isinstance(self.grids, dict) or len(self.grids.keys()) == 0:
            auto_disable = bool(
                _cfg_get(mc, "auto_disable_oracle_deployable_if_no_grids", False)
            )
            _log("[GRIDS][CHECK] expanded grids are EMPTY.")
            _log(
                f"[GRIDS][CHECK] auto_disable_oracle_deployable_if_no_grids={auto_disable}"
            )
            if auto_disable:
                _log(
                    "[GRIDS][CHECK] AUTO-DISABLING oracle/deployable to continue (NOT a ceiling benchmark)."
                )
                return False, False
            raise RuntimeError(
                "[TUNING CONFIG ERROR] Grids are empty (0 keys). You enabled oracle/deployable, so tuning grids are REQUIRED.\n"
                "Fix:\n  - Provide method-grid in cfg.tuners/cfg.grids, OR\n"
                "  - Provide param-grid and let mc.py wrap it, OR\n"
                "  - Disable oracle/deployable."
            )

        self._grid_key_for_method = _build_method_alias_map(methods, self.grids)
        _log("[GRIDS][ALIAS] cfg.methods -> grids keys mapping (first 30):")
        for m in methods[:30]:
            _log(f"[GRIDS][ALIAS]   {m} -> {self._grid_key(m)}")

        missing: List[str] = []
        empty: List[str] = []

        for m in methods:
            key = self._grid_key(m)
            if key not in self.grids:
                missing.append(m)
                continue
            g = self.grids.get(key, None)
            if not isinstance(g, list) or len(g) == 0:
                empty.append(m)

        if missing or empty:
            available = sorted(list(self.grids.keys()))[:120]
            raise RuntimeError(
                "[TUNING CONFIG ERROR] Ceiling benchmark requires tuning grids for ALL methods in cfg.methods.\n"
                f"Missing methods in grids: {missing}\n"
                f"Empty grids: {empty}\n\n"
                "Context:\n"
                f"  cfg.methods = {methods}\n"
                f"  resolved grid keys = {[self._grid_key(m) for m in methods]}\n"
                f"  available grids keys (first 120) = {available}\n\n"
                "DIAG:\n"
                "  - If available keys look like kf_q/pll_kp/... then your grids are PARAM-GRID.\n"
                "  - This mc.py version auto-wraps param-grid. If you still see this error, your wrap produced empty grids.\n"
            )

        _log(
            "[GRIDS][CHECK] OK: grids exist for all methods required by ceiling benchmark."
        )
        return enable_oracle, enable_deployable

    # ----------------------------
    # Scenario utilities
    # ----------------------------

    def _scenario_names_from_signals(self, signals) -> List[str]:
        return sorted(list(signals.keys()))

    def _scenario_T(self) -> float:
        sc = getattr(self.cfg, "scenario", None)
        if isinstance(sc, dict) and "T" in sc:
            return float(sc["T"])
        return 5.0

    def _regen_scenario(self, scenario_id: str, seed: int):
        fs_phys = float(getattr(self.cfg, "fs_physics_hz", self.cfg.fs_dsp_hz))
        T = float(self._scenario_T())
        return get_test_signal(scenario_id=scenario_id, fs=fs_phys, T=T, seed=int(seed))

    # ----------------------------
    # MC perturbation (with rich stats)
    # ----------------------------

    def _perturb_voltage(
        self, v: np.ndarray, rng: np.random.Generator, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        sc_id = str(meta.get("scenario_id", ""))
        mc = getattr(self.cfg, "mc", None)
        if not mc:
            return np.asarray(v, dtype=float).copy(), {
                "clip_frac": 0.0,
                "timing_shift_samples": 0.0,
            }

        pert_raw = _cfg_get(mc, "perturb", None)
        pert = _normalize_perturb_keys(pert_raw)
        if not pert:
            return np.asarray(v, dtype=float).copy(), {
                "clip_frac": 0.0,
                "timing_shift_samples": 0.0,
            }

        out = np.asarray(v, dtype=float).copy()
        n = out.size
        if n == 0:
            return out, {"clip_frac": 0.0, "timing_shift_samples": 0.0}

        stats: Dict[str, float] = {
            "clip_frac": 0.0,
            "timing_shift_samples": 0.0,
            "amp_gain": 1.0,
            "noise_std": 0.0,
            "snr_db_effective": float("nan"),
            "harm_a3": 0.0,
            "harm_a5": 0.0,
            "harm_a7": 0.0,
            "impulses_count": 0.0,
        }

        sig_rms = float(np.sqrt(np.mean(out * out) + 1e-12))

        # Timing shift
        K_shift = int(_cfg_get(pert, "timing_samples_jitter", 0))
        if K_shift > 0:
            s = int(rng.integers(-K_shift, K_shift + 1))
            if s != 0:
                out = np.roll(out, s)
            stats["timing_shift_samples"] = float(s)

        # Gain jitter
        amp_std = float(_cfg_get(pert, "amp_pct_jitter", 0.0))
        amp_cap = float(_cfg_get(pert, "amp_cap_pct", 0.05))
        if amp_std > 0:
            g = float(1.0 + rng.normal(0.0, amp_std))
            g = float(np.clip(g, 1.0 - amp_cap, 1.0 + amp_cap))
            out *= g
            stats["amp_gain"] = float(g)

        # Noise
        add_awgn = "Pure" not in sc_id
        if add_awgn:
            noise_rms_pct = _cfg_get(pert, "noise_rms_pct", None)
            if noise_rms_pct is not None:
                noise_rms_pct = float(noise_rms_pct)
                noise_rms_cap = float(_cfg_get(pert, "noise_rms_cap_pct", 0.03))
                nr = float(np.clip(noise_rms_pct, 0.0, noise_rms_cap))
                noise_std = nr * sig_rms
                if noise_std > 0:
                    out += rng.normal(0.0, noise_std, size=out.shape)
                stats["noise_std"] = float(noise_std)
            else:
                snr_jit = _cfg_get(pert, "snr_db_jitter", None)
                if snr_jit is not None and float(snr_jit) > 0:
                    snr_base = float(_cfg_get(pert, "snr_db", 40.0))
                    snr_db = float(snr_base + rng.normal(0.0, float(snr_jit)))
                    snr_db = float(np.clip(snr_db, 10.0, 80.0))
                    noise_std = sig_rms / (10.0 ** (snr_db / 20.0))
                    noise_std = float(np.clip(noise_std, 0.0, 0.03 * sig_rms))
                    if noise_std > 0:
                        out += rng.normal(0.0, noise_std, size=out.shape)
                    stats["noise_std"] = float(noise_std)
                    stats["snr_db_effective"] = float(snr_db)

        # Harmonics
        harm_jit = float(_cfg_get(pert, "harmonics_pct_jitter", 0.0))
        if harm_jit > 0:
            fs = float(getattr(self.cfg, "fs_dsp_hz", 1.0))
            f0 = float(_cfg_get(pert, "nominal_f_hz", 60.0))
            t = np.arange(n, dtype=float) / max(fs, 1.0)
            a3 = float(np.clip(abs(rng.normal(0.0, harm_jit)), 0.0, 0.10))
            a5 = float(np.clip(abs(rng.normal(0.0, harm_jit * 0.7)), 0.0, 0.07))
            a7 = float(np.clip(abs(rng.normal(0.0, harm_jit * 0.5)), 0.0, 0.05))
            p3 = float(rng.uniform(0.0, 2.0 * np.pi))
            p5 = float(rng.uniform(0.0, 2.0 * np.pi))
            p7 = float(rng.uniform(0.0, 2.0 * np.pi))
            harm = (
                (a3 * sig_rms) * np.sin(2.0 * np.pi * (3.0 * f0) * t + p3)
                + (a5 * sig_rms) * np.sin(2.0 * np.pi * (5.0 * f0) * t + p5)
                + (a7 * sig_rms) * np.sin(2.0 * np.pi * (7.0 * f0) * t + p7)
            )
            out = out + harm
            stats["harm_a3"] = float(a3)
            stats["harm_a5"] = float(a5)
            stats["harm_a7"] = float(a7)

        # Impulses
        allow_impulses_in = _cfg_get(pert, "allow_impulses_in", None)
        if not isinstance(allow_impulses_in, (list, tuple)):
            allow_impulses_in = ["G3_E13_Impulsive_Outliers"]
        if sc_id in set(map(str, allow_impulses_in)):
            p_imp = float(_cfg_get(pert, "impulsive_prob", 0.0))
            if p_imp > 0:
                scale_pct = _cfg_get(pert, "impulsive_scale_pct", None)
                if scale_pct is not None:
                    scale_pct = float(scale_pct)
                else:
                    imp_scale = _cfg_get(pert, "impulsive_scale", 1.0)
                    scale_pct = float(imp_scale) / 100.0
                scale_pct = float(np.clip(scale_pct, 0.0, 0.20))
                cap_pct = float(_cfg_get(pert, "impulsive_cap_pct", 0.05))
                cap_pct = float(np.clip(cap_pct, 0.0, 0.50))
                mask = rng.random(size=out.shape) < p_imp
                k = int(np.sum(mask))
                if k > 0:
                    spikes = rng.normal(0.0, scale_pct * sig_rms, size=k)
                    lim2 = cap_pct * sig_rms
                    spikes = np.clip(spikes, -lim2, +lim2)
                    out[mask] += spikes
                stats["impulses_count"] = float(k)

        # Hard clip
        hard_cap = float(_cfg_get(pert, "hard_clip_rms_mult", 6.0))
        lim = hard_cap * sig_rms
        before = out.copy()
        out = np.clip(out, -lim, +lim)
        clip_frac = float(np.mean(before != out)) if out.size > 0 else 0.0
        stats["clip_frac"] = float(clip_frac)

        return out, stats

    # ============================================================
    # Stage 0: Plan
    # ============================================================

    def _make_plan(
        self, scenario_names: List[str], methods: List[str], mc: Dict[str, Any]
    ) -> Dict[str, Any]:
        train_scenarios = _cfg_get(mc, "train_scenarios", None)
        test_scenarios = _cfg_get(mc, "test_scenarios", None)
        split_test_frac = float(_cfg_get(mc, "split_test_frac", 0.30))

        if isinstance(train_scenarios, (list, tuple)) and len(train_scenarios) > 0:
            S_train = [s for s in scenario_names if s in set(map(str, train_scenarios))]
            S_test = [s for s in scenario_names if s not in set(S_train)]
        elif isinstance(test_scenarios, (list, tuple)) and len(test_scenarios) > 0:
            S_test = [s for s in scenario_names if s in set(map(str, test_scenarios))]
            S_train = [s for s in scenario_names if s not in set(S_test)]
        else:
            ranked = sorted(scenario_names, key=lambda s: _stable_hash(s))
            n_test = max(1, int(round(len(ranked) * split_test_frac)))
            S_test = ranked[:n_test]
            S_train = ranked[n_test:]

        n_train = int(mc["n_train_seeds"])
        n_test = int(mc["n_test_seeds"])
        base_seed = int(mc["base_seed"])
        train_seeds = [base_seed + i for i in range(n_train)]
        test_seeds = [base_seed + n_train + i for i in range(n_test)]

        enable_oracle = bool(
            _cfg_get(mc, "enable_oracle", _cfg_get(mc, "enable_oracle_mode", True))
        )
        enable_deployable = bool(
            _cfg_get(
                mc, "enable_deployable", _cfg_get(mc, "enable_deployable_mode", True)
            )
        )

        enable_sens = bool(_cfg_get(mc, "enable_sensitivity", True))
        deltas = [
            float(x)
            for x in _cfg_get(mc, "sensitivity_deltas", [0.01, 0.05, 0.10])
            if float(x) > 0
        ]
        K = int(_cfg_get(mc, "sensitivity_k", 30))
        sens_seed_off = int(_cfg_get(mc, "sensitivity_seed_offset", 1000000))

        tune_frac_legacy = float(_cfg_get(mc, "tune_frac", 1.0))
        tune_frac_global = float(_cfg_get(mc, "tune_frac_global", tune_frac_legacy))
        tune_frac_oracle = float(_cfg_get(mc, "tune_frac_oracle", tune_frac_legacy))

        plan = {
            "scenarios_all": list(scenario_names),
            "scenarios_train": list(S_train),
            "scenarios_test": list(S_test),
            "methods": list(methods),
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "enable_oracle": enable_oracle,
            "enable_deployable": enable_deployable,
            "enable_sensitivity": enable_sens,
            "sensitivity_deltas": deltas,
            "sensitivity_k": K,
            "sensitivity_seed_offset": sens_seed_off,
            "tune_frac": float(tune_frac_legacy),
            "tune_frac_global": float(tune_frac_global),
            "tune_frac_oracle": float(tune_frac_oracle),
            "downsampling_ratio": int(self.cfg.downsampling_ratio),
            "fs_dsp_hz": float(self.cfg.fs_dsp_hz),
            "fs_physics_hz": float(
                getattr(self.cfg, "fs_physics_hz", self.cfg.fs_dsp_hz)
            ),
            "T": float(self._scenario_T()),
            # reproducibility / anti-leak
            "tuning_dataset_hash": _hash_dataset(
                {
                    "scenarios_train": list(S_train),
                    "train_seeds": train_seeds,
                    "tune_frac_global": float(tune_frac_global),
                    "tune_frac_oracle": float(tune_frac_oracle),
                    "methods": list(methods),
                }
            ),
            "evaluation_dataset_hash": _hash_dataset(
                {
                    "scenarios_eval": list(scenario_names),
                    "test_seeds": test_seeds,
                    "methods": list(methods),
                }
            ),
        }
        return plan

    # ============================================================
    # Tuning helpers
    # ============================================================

    def _prepare_fit_eval(
        self, sc_name: str, seed: int, tune_frac: float, rng: np.random.Generator
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[str, Any],
        Dict[str, float],
    ]:
        t_phys, v_ana, f_true, meta = self._regen_scenario(sc_name, seed=seed)
        ratio = int(self.cfg.downsampling_ratio)
        v_ds, f_ds, t_ds = v_ana[::ratio], f_true[::ratio], t_phys[::ratio]
        v_noisy, pert_stats = self._perturb_voltage(v_ds, rng, meta)

        cut = int(round(len(v_ds) * max(0.0, min(1.0, float(tune_frac)))))
        idx_fit = np.arange(0, max(1, cut))

        v_fit, f_fit = v_noisy[idx_fit], f_ds[idx_fit]
        v_eval, f_eval, t_eval = v_noisy, f_ds, t_ds
        return v_fit, f_fit, v_eval, f_eval, t_eval, meta, pert_stats

    def _score_rmse(self, trace: np.ndarray, f_ref: np.ndarray) -> float:
        L = int(min(len(trace), len(f_ref)))
        if L <= 0:
            return 1e9
        e = trace[:L] - f_ref[:L]
        rmse = float(np.sqrt(np.mean(e * e)))
        return rmse if np.isfinite(rmse) else 1e9

    def _eval_theta_on_fitset(
        self,
        make_est,
        base_params: Dict[str, Any],
        theta: Dict[str, Any],
        fitset: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        rmses: List[float] = []
        for v_fit, f_fit in fitset:
            est = make_est({**base_params, **theta})
            if hasattr(est, "reset"):
                est.reset()
            f_hat = np.array([est.step(float(x)) for x in v_fit], dtype=float)
            rmses.append(self._score_rmse(f_hat, f_fit))
        return float(np.mean(rmses)) if rmses else 1e9

    def _tune_grid(
        self,
        method: str,
        make_est,
        grid: List[Dict[str, Any]],
        base_params: Dict[str, Any],
        fitset: List[Tuple[np.ndarray, np.ndarray]],
        *,
        stage: str,
        scenario: Optional[str],
        pbar: Optional[tqdm] = None,
    ) -> Dict[str, Any]:
        if not isinstance(grid, list) or len(grid) == 0:
            raise RuntimeError(
                f"[TUNING FAIL] grid is empty. method={method}, stage={stage}, scenario={scenario}"
            )

        best_theta: Dict[str, Any] = {}
        best_score = float("inf")

        n_eval = 0
        for theta_raw in grid:
            theta_raw = dict(theta_raw) if theta_raw else {}
            theta = _sanitize_theta_strict(
                theta_raw, method=method, stage=stage, scenario=scenario
            )

            try:
                score = self._eval_theta_on_fitset(make_est, base_params, theta, fitset)
            except Exception as e:
                print(
                    "\n[TUNING CRASH] estimator failed during θ evaluation.\n"
                    f"method={method}, stage={stage}, scenario={scenario}\n"
                    f"theta={theta}\n"
                    f"error={type(e).__name__}: {e}\n",
                    file=sys.stderr,
                )
                raise

            n_eval += 1
            if score < best_score:
                best_score = score
                best_theta = dict(theta)

            if pbar is not None:
                pbar.update(1)

        if n_eval == 0:
            raise RuntimeError(
                f"[TUNING FAIL] No thetas evaluated at all. method={method}, stage={stage}, scenario={scenario}"
            )
        if best_theta == {}:
            raise RuntimeError(
                f"[TUNING FAIL] No usable theta found (best_theta empty). method={method}, stage={stage}, scenario={scenario}"
            )
        return best_theta

    # ============================================================
    # Stage 1: ORACLE tuning
    # ============================================================

    def _stage_tune_oracle(
        self, plan: Dict[str, Any]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        if not plan["enable_oracle"]:
            _log("[TUNE][ORACLE] disabled")
            return {}

        methods = plan["methods"]
        S_oracle = plan["scenarios_all"]
        train_seeds = plan["train_seeds"]
        tune_frac = float(plan.get("tune_frac_oracle", plan["tune_frac"]))

        _log(
            f"[TUNE][ORACLE] start: scenarios={len(S_oracle)} methods={len(methods)} train_seeds={len(train_seeds)} tune_frac={tune_frac}"
        )

        theta_oracle: Dict[Tuple[str, str], Dict[str, Any]] = {}

        total = 0
        for m in methods:
            total += len(self._grid(m)) * len(S_oracle)

        _log(f"[TUNE][ORACLE] total θ-evals={total}")
        if total <= 0:
            raise RuntimeError(
                "[TUNING FAIL] Stage 1 total=0 θ-evals. This means grids are empty / not loaded."
            )

        pbar = tqdm(total=total, desc="Stage 1/5: Tuning ORACLE", unit="θ-eval")

        try:
            for sc in S_oracle:
                _log(f"[TUNE][ORACLE] scenario={sc}")
                for m in methods:
                    g_len = len(self._grid(m))
                    _log(f"[TUNE][ORACLE]   method={m} grid_len={g_len}")
                    spec = self.registry.get(m)
                    make_est = spec.builder
                    base_params = {"fs_hz": float(plan["fs_dsp_hz"])}

                    fitset: List[Tuple[np.ndarray, np.ndarray]] = []
                    for sd in train_seeds:
                        rng = np.random.default_rng(int(sd))
                        v_fit, f_fit, _, _, _, _, _ = self._prepare_fit_eval(
                            sc, sd, tune_frac, rng
                        )
                        fitset.append((v_fit, f_fit))

                    best_theta = self._tune_grid(
                        method=m,
                        make_est=make_est,
                        grid=self._grid(m),
                        base_params=base_params,
                        fitset=fitset,
                        stage="oracle",
                        scenario=sc,
                        pbar=pbar,
                    )
                    theta_oracle[(m, sc)] = best_theta

            save_json(
                os.path.join(self.tuning_dir, "theta_oracle.json"),
                {
                    "type": "oracle",
                    "scenarios_oracle": S_oracle,
                    "train_seeds": train_seeds,
                    "methods": methods,
                    "tune_frac_oracle": tune_frac,
                    "theta": [
                        {"method": m, "scenario": sc, "theta": th}
                        for (m, sc), th in theta_oracle.items()
                    ],
                },
            )
            _log("[TUNE][ORACLE] saved=artifacts/results_mc/tuning/theta_oracle.json")
            return theta_oracle
        finally:
            pbar.close()

    # ============================================================
    # Stage 2: GLOBAL tuning
    # ============================================================

    def _stage_tune_global(self, plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not plan["enable_deployable"]:
            _log("[TUNE][GLOBAL] disabled")
            return {}

        methods = plan["methods"]
        S_train = plan["scenarios_train"]
        train_seeds = plan["train_seeds"]
        tune_frac = float(plan.get("tune_frac_global", plan["tune_frac"]))

        _log(
            f"[TUNE][GLOBAL] start: scenarios_train={len(S_train)} methods={len(methods)} train_seeds={len(train_seeds)} tune_frac={tune_frac}"
        )

        theta_global: Dict[str, Dict[str, Any]] = {}

        total = 0
        for m in methods:
            total += len(self._grid(m))

        _log(f"[TUNE][GLOBAL] total θ-evals={total}")
        if total <= 0:
            raise RuntimeError(
                "[TUNING FAIL] Stage 2 total=0 θ-evals. This means grids are empty / not loaded."
            )

        pbar = tqdm(total=total, desc="Stage 2/5: Tuning GLOBAL", unit="θ-eval")

        try:
            for m in methods:
                g_len = len(self._grid(m))
                _log(
                    f"[TUNE][GLOBAL] method={m} grid_len={g_len} scenarios_train={len(S_train)}"
                )
                spec = self.registry.get(m)
                make_est = spec.builder
                base_params = {"fs_hz": float(plan["fs_dsp_hz"])}

                fitset: List[Tuple[np.ndarray, np.ndarray]] = []
                for sc in S_train:
                    for sd in train_seeds:
                        rng = np.random.default_rng(int(sd))
                        v_fit, f_fit, _, _, _, _, _ = self._prepare_fit_eval(
                            sc, sd, tune_frac, rng
                        )
                        fitset.append((v_fit, f_fit))

                best_theta = self._tune_grid(
                    method=m,
                    make_est=make_est,
                    grid=self._grid(m),
                    base_params=base_params,
                    fitset=fitset,
                    stage="global",
                    scenario=None,
                    pbar=pbar,
                )
                theta_global[m] = best_theta

            save_json(
                os.path.join(self.tuning_dir, "theta_global.json"),
                {
                    "type": "global",
                    "scenarios_train": S_train,
                    "train_seeds": train_seeds,
                    "methods": methods,
                    "tune_frac_global": tune_frac,
                    "theta": [
                        {"method": m, "theta": th} for m, th in theta_global.items()
                    ],
                },
            )
            _log("[TUNE][GLOBAL] saved=artifacts/results_mc/tuning/theta_global.json")
            return theta_global
        finally:
            pbar.close()

    # ============================================================
    # Sensitivity: perturb θ
    # ============================================================

    def _perturb_theta(
        self, theta: Dict[str, Any], delta_pct: float, rng: np.random.Generator
    ) -> Dict[str, Any]:
        if not theta or delta_pct <= 0:
            return dict(theta)
        out: Dict[str, Any] = {}
        for k, v in theta.items():
            if k == "fs_hz":
                out[k] = v
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(
                float(v)
            ):
                x = float(v)
                u = float(rng.uniform(-delta_pct, +delta_pct))
                if abs(x) > 1e-10:
                    out[k] = float(x * (1.0 + u))
                else:
                    scale = max(1e-6, abs(x) + 1e-6)
                    out[k] = float(x + u * scale)
            else:
                out[k] = v
        return out

    # ============================================================
    # Build jobs
    # ============================================================

    def _build_baseline_jobs(
        self,
        plan: Dict[str, Any],
        theta_oracle: Dict[Tuple[str, str], Dict[str, Any]],
        theta_global: Dict[str, Dict[str, Any]],
    ) -> List[RunJob]:
        jobs: List[RunJob] = []
        S_eval = plan["scenarios_all"]
        seeds = plan["test_seeds"]
        methods = plan["methods"]
        enable_oracle = plan["enable_oracle"]
        enable_deployable = plan["enable_deployable"]

        mc = getattr(self.cfg, "mc", {}) or {}

        export_policy = str(
            _cfg_get(mc, "export_waveforms_seed_policy", "first_test_seed")
        ).lower()
        if export_policy == "none":
            export_seed: Optional[int] = None
        elif export_policy == "fixed":
            export_seed = int(
                _cfg_get(mc, "export_waveforms_seed_fixed", seeds[0] if seeds else 0)
            )
        else:
            export_seed = seeds[0] if len(seeds) > 0 else None

        export_modes = _cfg_get(mc, "export_waveforms_modes", ["deployable", "oracle"])
        if not isinstance(export_modes, (list, tuple)):
            export_modes = ["deployable"]
        export_modes_set = set(map(str, export_modes))

        export_methods = _cfg_get(mc, "export_waveforms_methods", None)
        export_methods_set = (
            set(map(str, export_methods))
            if isinstance(export_methods, (list, tuple))
            else None
        )

        export_scenarios = _cfg_get(mc, "export_waveforms_scenarios", None)
        export_scenarios_set = (
            set(map(str, export_scenarios))
            if isinstance(export_scenarios, (list, tuple))
            else None
        )

        limit_methods = int(
            _cfg_get(mc, "export_waveforms_limit_methods", 0)
        )  # 0=unlimited

        for sd in seeds:
            for sc in S_eval:
                for m in methods:
                    export_this_seed = export_seed is not None and int(sd) == int(
                        export_seed
                    )
                    export_this_scenario = (
                        export_scenarios_set is None or sc in export_scenarios_set
                    )
                    export_this_method = (
                        export_methods_set is None or m in export_methods_set
                    )

                    if limit_methods > 0:
                        allowed = set(sorted(methods)[:limit_methods])
                        export_this_method = export_this_method and (m in allowed)

                    if enable_deployable:
                        jobs.append(
                            RunJob(
                                stage="baseline",
                                scenario=sc,
                                seed=int(sd),
                                seed_role="test",
                                method=m,
                                mode="deployable",
                                theta=dict(theta_global.get(m, {})),
                                delta_pct=0.0,
                                perturb_id=None,
                                export_waveforms=bool(
                                    export_this_seed
                                    and export_this_scenario
                                    and export_this_method
                                    and ("deployable" in export_modes_set)
                                ),
                            )
                        )
                    if enable_oracle:
                        th = dict(theta_oracle.get((m, sc), {}))
                        if th == {}:
                            raise RuntimeError(
                                f"[TUNING FAIL] Missing oracle theta for method={m}, scenario={sc}. "
                                "Ceiling benchmark requires oracle theta for every (scenario, method)."
                            )
                        jobs.append(
                            RunJob(
                                stage="baseline",
                                scenario=sc,
                                seed=int(sd),
                                seed_role="test",
                                method=m,
                                mode="oracle",
                                theta=th,
                                delta_pct=0.0,
                                perturb_id=None,
                                export_waveforms=bool(
                                    export_this_seed
                                    and export_this_scenario
                                    and export_this_method
                                    and ("oracle" in export_modes_set)
                                ),
                            )
                        )
        return jobs

    def _build_sensitivity_jobs(
        self, plan: Dict[str, Any], baseline_jobs: List[RunJob]
    ) -> List[RunJob]:
        jobs: List[RunJob] = []
        if not plan["enable_sensitivity"]:
            return jobs

        deltas = list(plan["sensitivity_deltas"])
        K = int(plan["sensitivity_k"])
        sens_off = int(plan["sensitivity_seed_offset"])

        for bj in baseline_jobs:
            for d in deltas:
                for k in range(K):
                    key = f"{bj.scenario}|{bj.seed}|{bj.method}|{bj.mode}|{d}|{k}|{sens_off}"
                    pert_seed = _stable_hash(key) % (2**31 - 1)
                    rng = np.random.default_rng(int(pert_seed))
                    thp = self._perturb_theta(bj.theta, float(d), rng)
                    jobs.append(
                        RunJob(
                            stage="sensitivity",
                            scenario=bj.scenario,
                            seed=int(bj.seed),
                            seed_role=bj.seed_role,
                            method=bj.method,
                            mode=bj.mode,
                            theta=thp,
                            delta_pct=float(d),
                            perturb_id=int(k),
                            export_waveforms=False,
                        )
                    )
        return jobs

    # ============================================================
    # Serial job execution (waveform export correctness)
    # ============================================================

    def _run_job_serial(self, job: RunJob, mc_run_id: str) -> Dict[str, Any]:
        sc_name = job.scenario
        seed = int(job.seed)
        rng = np.random.default_rng(seed)

        spec = self.registry.get(job.method)
        method_family = str(getattr(spec, "family", "unknown"))

        try:
            t_phys, v_ana, f_true, meta = self._regen_scenario(sc_name, seed=seed)
            ratio = int(self.cfg.downsampling_ratio)
            v_ds, f_ds, t_ds = v_ana[::ratio], f_true[::ratio], t_phys[::ratio]
            v_eval, pert_stats = self._perturb_voltage(v_ds, rng, meta)
            f_eval, t_eval = f_ds, t_ds
        except Exception as e:
            pair_id = _job_id(
                {
                    "stage": job.stage,
                    "scenario": job.scenario,
                    "seed": seed,
                    "seed_role": job.seed_role,
                    "mode": job.mode,
                    "delta_pct": float(job.delta_pct),
                    "perturb_id": (
                        None if job.perturb_id is None else int(job.perturb_id)
                    ),
                }
            )
            return {
                "mc_run_id": mc_run_id,
                "pair_id": pair_id,
                "job_id": _job_id(job.to_dict()),
                "stage": job.stage,
                "scenario": sc_name,
                "scenario_id": sc_name,
                "seed": seed,
                "seed_role": job.seed_role,
                "method": job.method,
                "method_family": method_family,
                "mode": job.mode,
                "delta_pct": float(job.delta_pct),
                "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
                "fs_dsp_hz": float(self.cfg.fs_dsp_hz),
                "downsampling_ratio": int(self.cfg.downsampling_ratio),
                "latency_samples": 0,
                "exec_time_s": float("nan"),
                "clipped_frac": 0.0,
                "nan_frac": 1.0,
                "valid_trace": False,
                "hard_fail": True,
                "theta": job.theta,
                "_error": f"scenario_gen_failed: {type(e).__name__}: {e}",
            }

        make_est = spec.builder
        base_params = {"fs_hz": float(self.cfg.fs_dsp_hz)}

        t_start = time.perf_counter()
        err = None
        try:
            est = make_est({**base_params, **(job.theta or {})})
            if hasattr(est, "reset"):
                est.reset()
            f_hat = np.array([est.step(float(x)) for x in v_eval], dtype=float)
        except Exception as e:
            err = e
            f_hat = np.full_like(f_eval, np.nan, dtype=float)
        exec_t = float(time.perf_counter() - t_start)

        latency = 0
        try:
            if getattr(spec, "structural_latency", None) is not None:
                latency = int(spec.structural_latency(est))  # type: ignore[name-defined]
        except Exception:
            latency = 0

        # RAW metrics
        try:
            mets_raw = compute_metrics(
                f_hat, f_eval, exec_t, latency, self.metric_cfg, sc_name
            )
        except Exception as e:
            mets_raw = {"_metrics_error": f"{type(e).__name__}: {e}"}

        # ALIGNED metrics
        try:
            f_hat_a, f_eval_a = _align_by_latency(f_hat, f_eval, latency)
            mets_al = compute_metrics(
                f_hat_a, f_eval_a, exec_t, 0, self.metric_cfg, sc_name
            )
        except Exception as e:
            mets_al = {"_metrics_error": f"{type(e).__name__}: {e}"}

        nan_frac = float(np.mean(~np.isfinite(f_hat))) if f_hat.size > 0 else 1.0
        valid_trace = bool(
            np.all(np.isfinite(f_hat)) and f_hat.size == f_eval.size and f_hat.size > 0
        )
        hard_fail = bool((not valid_trace) or (err is not None))

        scenario_id = str(meta.get("scenario_id", sc_name))
        pair_id = _job_id(
            {
                "stage": job.stage,
                "scenario": job.scenario,
                "seed": seed,
                "seed_role": job.seed_role,
                "mode": job.mode,
                "delta_pct": float(job.delta_pct),
                "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
            }
        )

        fs_dsp = float(self.cfg.fs_dsp_hz)
        wup = _warmup_samples(fs_dsp)
        Ltot = int(min(np.asarray(f_hat).size, np.asarray(f_eval).size))
        wup_eff = int(wup if Ltot > wup else 0)
        eval_samples = int(max(0, Ltot - wup_eff))

        if job.export_waveforms and (job.delta_pct == 0.0) and (job.perturb_id is None):
            self._export_q1_data(
                sc_name=sc_name,
                t=t_eval,
                v=v_eval,
                f_true=f_eval,
                scenario_meta=meta,
                m_name=f"{job.method}__{job.mode}",
                f_hat=f_hat,
                tuning=job.theta,
                latency=latency,
                exec_t=exec_t,
                mets=mets_raw,
                perturb_stats=pert_stats,
            )

        rec: Dict[str, Any] = {
            "mc_run_id": mc_run_id,
            "pair_id": pair_id,
            "job_id": _job_id(job.to_dict()),
            "stage": job.stage,
            "scenario": sc_name,
            "scenario_id": scenario_id,
            "seed": seed,
            "seed_role": job.seed_role,
            "method": job.method,
            "method_family": method_family,
            "mode": job.mode,
            "delta_pct": float(job.delta_pct),
            "perturb_id": None if job.perturb_id is None else int(job.perturb_id),
            "fs_dsp_hz": fs_dsp,
            "downsampling_ratio": int(self.cfg.downsampling_ratio),
            "latency_samples": int(latency),
            "effective_latency_s": float(latency) / max(fs_dsp, 1.0),
            "alignment_policy": "shift_fhat_by_latency_then_crop",
            "warm_up_samples": int(wup_eff),
            "eval_samples": int(eval_samples),
            "eval_duration_s": float(eval_samples) / max(fs_dsp, 1.0),
            # perturbation stats
            "clipped_frac": float(pert_stats.get("clip_frac", 0.0)),
            "pert_timing_shift_samples": float(
                pert_stats.get("timing_shift_samples", 0.0)
            ),
            "pert_amp_gain": float(pert_stats.get("amp_gain", 1.0)),
            "pert_noise_std": float(pert_stats.get("noise_std", 0.0)),
            "pert_snr_db_effective": float(
                pert_stats.get("snr_db_effective", float("nan"))
            ),
            "pert_harm_a3": float(pert_stats.get("harm_a3", 0.0)),
            "pert_harm_a5": float(pert_stats.get("harm_a5", 0.0)),
            "pert_harm_a7": float(pert_stats.get("harm_a7", 0.0)),
            "pert_impulses_count": float(pert_stats.get("impulses_count", 0.0)),
            "exec_time_s": float(exec_t),
            "nan_frac": float(nan_frac),
            "valid_trace": bool(valid_trace),
            "hard_fail": bool(hard_fail),
            "theta": job.theta,
        }
        if err is not None:
            rec["_error"] = f"estimator_failed: {type(err).__name__}: {err}"

        # RAW
        for k, v in mets_raw.items():
            num = _metric_obj_to_float(v)
            if num is not None:
                rec[k] = float(num)

        # ALIGNED (skip trip-time + compute cost)
        skip_prefixes = ("TRIP_TIME_", "TIME_PER_SAMPLE_US", "LATENCY_SAMPLES")
        for k, v in mets_al.items():
            if any(str(k).startswith(p) for p in skip_prefixes):
                continue
            if str(k).startswith("_"):
                continue
            num = _metric_obj_to_float(v)
            if num is not None:
                rec[f"{k}_ALIGNED"] = float(num)

        return rec

    # ============================================================
    # Execute jobs (serial or parallel-safe)
    # ============================================================

    def _execute_jobs(
        self,
        jobs: List[RunJob],
        out_jsonl_path: str,
        desc: str,
        mc: Dict[str, Any],
        mc_run_id: str,
    ) -> None:
        if os.path.exists(out_jsonl_path):
            os.remove(out_jsonl_path)

        writer = JsonlWriter(out_jsonl_path, flush_every=200)
        pbar = tqdm(total=max(1, len(jobs)), desc=desc, unit="run")

        force_serial = bool(_cfg_get(mc, "force_serial_eval", False))

        can_parallel = False
        cfg_pkl = reg_pkl = metric_pkl = None
        if not force_serial:
            try:
                cfg_pkl = pickle.dumps(self.cfg)
                reg_pkl = pickle.dumps(self.registry)
                metric_pkl = pickle.dumps(self.metric_cfg)
                can_parallel = True
            except Exception:
                can_parallel = False

        export_jobs_idx = set(i for i, j in enumerate(jobs) if j.export_waveforms)

        try:
            if not can_parallel:
                _log(
                    f"[EXEC] {desc}: running SERIAL (can_parallel={can_parallel}, force_serial={force_serial}) jobs={len(jobs)}"
                )
                for job in jobs:
                    rec = self._run_job_serial(job, mc_run_id=mc_run_id)
                    writer.append(rec)
                    pbar.update(1)
                return

            from concurrent.futures import ProcessPoolExecutor, as_completed

            n_workers = int(_cfg_get(mc, "n_workers", 0))
            if n_workers <= 0:
                n_workers = max(1, (os.cpu_count() or 4) - 1)

            _log(
                f"[EXEC] {desc}: running PARALLEL workers={n_workers} jobs={len(jobs)} export_serial_jobs={len(export_jobs_idx)}"
            )

            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for i, job in enumerate(jobs):
                    if i in export_jobs_idx:
                        rec = self._run_job_serial(job, mc_run_id=mc_run_id)
                        writer.append(rec)
                        pbar.update(1)
                        continue

                    payload = {
                        "cfg_pkl": cfg_pkl,
                        "registry_pkl": reg_pkl,
                        "metric_cfg_pkl": metric_pkl,
                        "job": job.to_dict(),
                        "mc_run_id": mc_run_id,
                    }
                    futures[ex.submit(_worker_run_job, payload)] = job

                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        seed = int(job.seed)
                        pair_id = _job_id(
                            {
                                "stage": job.stage,
                                "scenario": job.scenario,
                                "seed": seed,
                                "seed_role": job.seed_role,
                                "mode": job.mode,
                                "delta_pct": float(job.delta_pct),
                                "perturb_id": (
                                    None
                                    if job.perturb_id is None
                                    else int(job.perturb_id)
                                ),
                            }
                        )
                        rec = {
                            "mc_run_id": mc_run_id,
                            "pair_id": pair_id,
                            "job_id": _job_id(job.to_dict()),
                            "stage": job.stage,
                            "scenario": job.scenario,
                            "scenario_id": job.scenario,
                            "seed": seed,
                            "seed_role": job.seed_role,
                            "method": job.method,
                            "method_family": "unknown",
                            "mode": job.mode,
                            "delta_pct": float(job.delta_pct),
                            "perturb_id": (
                                None if job.perturb_id is None else int(job.perturb_id)
                            ),
                            "fs_dsp_hz": float(self.cfg.fs_dsp_hz),
                            "downsampling_ratio": int(self.cfg.downsampling_ratio),
                            "latency_samples": 0,
                            "exec_time_s": float("nan"),
                            "clipped_frac": 0.0,
                            "nan_frac": 1.0,
                            "valid_trace": False,
                            "hard_fail": True,
                            "theta": job.theta,
                            "_error": f"worker_crashed: {type(e).__name__}: {e}",
                            "_traceback": traceback.format_exc(limit=10),
                        }
                    writer.append(rec)
                    pbar.update(1)

        finally:
            pbar.close()
            writer.close()

    # ============================================================
    # Summary from JSONL (with COVERAGE)
    # ============================================================

    def _read_jsonl_to_df(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def _summarize_df(self, df: pd.DataFrame, ps: List[int]) -> Dict[str, Any]:
        if df.empty:
            return {"results": {}}

        ignore = {
            "mc_run_id",
            "pair_id",
            "job_id",
            "stage",
            "scenario",
            "scenario_id",
            "seed",
            "seed_role",
            "method",
            "method_family",
            "mode",
            "delta_pct",
            "perturb_id",
            "theta",
            "alignment_policy",
        }
        num_cols = [
            c
            for c in df.columns
            if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
        ]

        out: Dict[str, Any] = {"results": {}}
        g = df.groupby(["scenario", "method", "mode"], dropna=False)

        for (sc, m, mode), block in g:
            sc, m, mode = str(sc), str(m), str(mode)
            tgt = (
                out["results"]
                .setdefault(sc, {})
                .setdefault("methods", {})
                .setdefault(m, {})
                .setdefault(mode, {})
            )

            n_runs = int(len(block))
            n_valid = (
                int(block["valid_trace"].astype(bool).sum())
                if "valid_trace" in block.columns
                else 0
            )
            n_hfail = (
                int(block["hard_fail"].astype(bool).sum())
                if "hard_fail" in block.columns
                else 0
            )

            tgt["_coverage"] = {
                "n_runs": n_runs,
                "n_valid": n_valid,
                "valid_rate": float(n_valid / max(1, n_runs)),
                "hard_fail_rate": float(n_hfail / max(1, n_runs)),
            }

            for col in num_cols:
                vals = (
                    pd.to_numeric(block[col], errors="coerce")
                    .dropna()
                    .to_numpy(dtype=float)
                )
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                tgt[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    **_percentiles(vals, ps),
                }

        return out

    def _export_runs_dataframe(
        self, df: pd.DataFrame, *, tag: str, mc: Dict[str, Any]
    ) -> None:
        if df.empty:
            return
        if not bool(_cfg_get(mc, "export_dataframe", True)):
            return

        path_parq = os.path.join(self.runs_dir, f"runs_{tag}.parquet")
        try:
            df.to_parquet(path_parq, index=False)
            _log(f"[EXPORT] runs_{tag}.parquet -> {path_parq}")
        except Exception as e:
            _log(f"[EXPORT] runs_{tag}.parquet failed: {type(e).__name__}: {e}")

        if bool(_cfg_get(mc, "export_dataframe_csv", False)):
            path_csv = os.path.join(self.runs_dir, f"runs_{tag}.csv")
            try:
                df.to_csv(path_csv, index=False)
                _log(f"[EXPORT] runs_{tag}.csv -> {path_csv}")
            except Exception as e:
                _log(f"[EXPORT] runs_{tag}.csv failed: {type(e).__name__}: {e}")

    # ============================================================
    # Timing strict pass (serial) for Q1
    # ============================================================

    def _timing_strict_pass(
        self,
        plan: Dict[str, Any],
        theta_oracle: Dict[Tuple[str, str], Dict[str, Any]],
        theta_global: Dict[str, Dict[str, Any]],
    ) -> None:
        mc = getattr(self.cfg, "mc", {}) or {}
        if not bool(_cfg_get(mc, "measure_timing_strict", False)):
            return

        methods = list(plan["methods"])
        if not methods:
            return
        sc0 = str(
            plan["scenarios_test"][0]
            if plan["scenarios_test"]
            else plan["scenarios_all"][0]
        )
        sd0 = int(
            plan["test_seeds"][0] if plan["test_seeds"] else plan["train_seeds"][0]
        )

        reps = int(_cfg_get(mc, "timing_reps", 3))
        reps = max(1, min(20, reps))

        rows: List[Dict[str, Any]] = []
        _log(
            f"[TIMING] strict pass (serial) reps={reps} scenario={sc0} seed={sd0} methods={len(methods)}"
        )

        # Fix perturbation randomness + avoid waveform export
        for m in methods:
            for mode in ("deployable", "oracle"):
                if mode == "deployable":
                    if not plan.get("enable_deployable", False):
                        continue
                    theta = dict(theta_global.get(m, {}))
                else:
                    if not plan.get("enable_oracle", False):
                        continue
                    theta = dict(theta_oracle.get((m, sc0), {}))

                if theta is None:
                    continue

                # repeat measurement to average noise
                tps_list = []
                lat_list = []
                for r in range(reps):
                    job = RunJob(
                        stage="timing",
                        scenario=sc0,
                        seed=int(sd0),
                        seed_role="test",
                        method=m,
                        mode=mode,
                        theta=theta,
                        delta_pct=0.0,
                        perturb_id=None,
                        export_waveforms=False,
                    )
                    rec = self._run_job_serial(
                        job, mc_run_id=plan.get("mc_run_id", "unknown")
                    )
                    tps = rec.get("TIME_PER_SAMPLE_US", None)
                    if isinstance(tps, (int, float)) and np.isfinite(float(tps)):
                        tps_list.append(float(tps))
                    lat_list.append(float(rec.get("latency_samples", 0.0)))

                row = {
                    "method": m,
                    "mode": mode,
                    "scenario": sc0,
                    "seed": sd0,
                    "reps": reps,
                    "time_per_sample_us_mean": (
                        float(np.mean(tps_list)) if tps_list else float("nan")
                    ),
                    "time_per_sample_us_std": (
                        float(np.std(tps_list)) if tps_list else float("nan")
                    ),
                    "latency_samples": (
                        float(np.mean(lat_list)) if lat_list else float("nan")
                    ),
                }
                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows).sort_values(["mode", "method"])
        path = os.path.join(self.exports_dir, "timing_strict.csv")
        try:
            df.to_csv(path, index=False)
            _log(f"[TIMING] exported -> {path}")
        except Exception as e:
            _log(f"[TIMING] export failed: {type(e).__name__}: {e}")

    # ============================================================
    # Public API
    # ============================================================

    def run(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        mc = getattr(self.cfg, "mc", None)
        if mc is None:
            raise ValueError("cfg.mc is required for MonteCarloRunner")

        for k in ("n_train_seeds", "n_test_seeds", "base_seed", "tune_frac"):
            if k not in mc:
                raise ValueError(f"cfg.mc missing required key: {k}")

        scenario_names = self._scenario_names_from_signals(signals)
        methods = list(getattr(self.cfg, "methods", []))
        if not methods:
            raise ValueError("cfg.methods is empty; nothing to run.")

        plan_preview = {
            "scenarios": scenario_names,
            "methods": methods,
            "mc": {k: mc.get(k) for k in sorted(mc.keys())},
            "fs_dsp_hz": float(self.cfg.fs_dsp_hz),
            "downsampling_ratio": int(self.cfg.downsampling_ratio),
        }
        mc_run_id = _job_id(plan_preview)

        _log("==============================================================")
        _log(f"[MC] run start mc_run_id={mc_run_id}")
        _log(
            f"[MC] fs_dsp_hz={float(self.cfg.fs_dsp_hz)} downsampling_ratio={int(self.cfg.downsampling_ratio)}"
        )
        _log(f"[MC] scenarios={len(scenario_names)} methods={len(methods)}")
        _log(f"[MC] methods={methods}")
        _log(f"[MC] mc keys: {', '.join(sorted(list(mc.keys())))}")
        _log("==============================================================")

        plan = self._make_plan(scenario_names, methods, mc)
        plan["mc_run_id"] = mc_run_id
        save_json(os.path.join(self.plan_dir, "plan.json"), plan)
        _log(f"[PLAN] saved={os.path.join(self.plan_dir, 'plan.json')}")
        _log(
            f"[PLAN] scenarios_all={len(plan['scenarios_all'])} train={len(plan['scenarios_train'])} test={len(plan['scenarios_test'])}"
        )
        _log(
            f"[PLAN] enable_oracle={plan['enable_oracle']} enable_deployable={plan['enable_deployable']} enable_sensitivity={plan['enable_sensitivity']}"
        )
        _log(
            f"[PLAN] tuning_dataset_hash={plan.get('tuning_dataset_hash')} evaluation_dataset_hash={plan.get('evaluation_dataset_hash')}"
        )

        plan["enable_oracle"], plan["enable_deployable"] = (
            self._assert_grids_for_all_methods(
                methods,
                enable_oracle=plan["enable_oracle"],
                enable_deployable=plan["enable_deployable"],
                mc=mc,
            )
        )

        theta_oracle = self._stage_tune_oracle(plan)
        theta_global = self._stage_tune_global(plan)

        baseline_jobs = self._build_baseline_jobs(plan, theta_oracle, theta_global)
        baseline_path = os.path.join(self.runs_dir, "runs_baseline.jsonl")
        _log(f"[BASELINE] jobs={len(baseline_jobs)} -> {baseline_path}")
        self._execute_jobs(
            jobs=baseline_jobs,
            out_jsonl_path=baseline_path,
            desc="Stage 3/5: Eval BASELINES",
            mc=mc,
            mc_run_id=mc_run_id,
        )

        sens_jobs = self._build_sensitivity_jobs(plan, baseline_jobs)
        sens_path = os.path.join(self.runs_dir, "runs_sensitivity.jsonl")
        if len(sens_jobs) > 0:
            _log(f"[SENS] jobs={len(sens_jobs)} -> {sens_path}")
            self._execute_jobs(
                jobs=sens_jobs,
                out_jsonl_path=sens_path,
                desc="Stage 4/5: Eval SENSITIVITY",
                mc=mc,
                mc_run_id=mc_run_id,
            )
        else:
            if os.path.exists(sens_path):
                os.remove(sens_path)

        ps = list(map(int, _cfg_get(mc, "report_percentiles", [1, 5, 50, 95, 99])))
        _log(f"[SUMMARY] percentiles={ps}")
        df_base = self._read_jsonl_to_df(baseline_path)
        df_sens = (
            self._read_jsonl_to_df(sens_path)
            if os.path.exists(sens_path)
            else pd.DataFrame()
        )

        self._export_runs_dataframe(df_base, tag="baseline", mc=mc)
        if not df_sens.empty:
            self._export_runs_dataframe(df_sens, tag="sensitivity", mc=mc)

        summary_base = self._summarize_df(df_base, ps=ps)
        summary_sens = (
            self._summarize_df(df_sens, ps=ps) if not df_sens.empty else {"results": {}}
        )

        # Optional strict timing pass (serial)
        self._timing_strict_pass(plan, theta_oracle, theta_global)

        final_res: Dict[str, Any] = {
            "mc_run_id": mc_run_id,
            "metadata": dict(mc),
            "plan": {
                "scenarios_all": plan["scenarios_all"],
                "scenarios_train": plan["scenarios_train"],
                "scenarios_test": plan["scenarios_test"],
                "train_seeds": plan["train_seeds"],
                "test_seeds": plan["test_seeds"],
                "enable_oracle": plan["enable_oracle"],
                "enable_deployable": plan["enable_deployable"],
                "enable_sensitivity": plan["enable_sensitivity"],
                "sensitivity_deltas": plan["sensitivity_deltas"],
                "sensitivity_k": plan["sensitivity_k"],
                "tune_frac": plan["tune_frac"],
                "tune_frac_global": plan.get("tune_frac_global", plan["tune_frac"]),
                "tune_frac_oracle": plan.get("tune_frac_oracle", plan["tune_frac"]),
                "tuning_dataset_hash": plan.get("tuning_dataset_hash"),
                "evaluation_dataset_hash": plan.get("evaluation_dataset_hash"),
            },
            "baseline": summary_base["results"],
            "sensitivity": summary_sens["results"],
            "artifacts": {
                "plan": os.path.join(self.plan_dir, "plan.json"),
                "theta_global": os.path.join(self.tuning_dir, "theta_global.json"),
                "theta_oracle": os.path.join(self.tuning_dir, "theta_oracle.json"),
                "runs_baseline": baseline_path,
                "runs_sensitivity": sens_path if os.path.exists(sens_path) else None,
                "runs_baseline_table": (
                    os.path.join(self.runs_dir, "runs_baseline.parquet")
                    if bool(_cfg_get(mc, "export_dataframe", True))
                    else None
                ),
                "runs_sensitivity_table": (
                    os.path.join(self.runs_dir, "runs_sensitivity.parquet")
                    if (
                        bool(_cfg_get(mc, "export_dataframe", True))
                        and os.path.exists(sens_path)
                    )
                    else None
                ),
                "cfg_snapshot": os.path.join(self.exports_dir, "cfg_snapshot.json"),
                "grids_snapshot": os.path.join(self.exports_dir, "grids_snapshot.json"),
                "methods_catalog": os.path.join(
                    self.exports_dir, "methods_catalog.json"
                ),
                "timing_strict": (
                    os.path.join(self.exports_dir, "timing_strict.csv")
                    if bool(_cfg_get(mc, "measure_timing_strict", False))
                    else None
                ),
            },
            "notes": {
                "ceiling_contract": (
                    "Oracle tuning is performed for every (scenario, method) using train_seeds; "
                    "baseline compares methods on all scenarios using test_seeds with the tuned theta*."
                ),
                "fail_fast_grid": (
                    "Tuning hard-fails if any theta still contains list/ndarray size>1 or if any method is missing a grid."
                ),
                "latency_fairness": (
                    "Exports both RAW and *_ALIGNED metrics; aligned metrics compensate estimator structural latency "
                    "by shifting f_hat forward by latency_samples and cropping to common length."
                ),
                "timing_warning": (
                    "exec_time_s is wall-clock and can be noisy under parallel eval; "
                    "timing_strict.csv is generated in serial mode if measure_timing_strict=True."
                ),
            },
        }

        save_json(os.path.join(self.out_dir, "mc_results.json"), final_res)
        _log(f"[MC] done -> artifacts/results_mc/mc_results.json")
        return final_res

    # ============================================================
    # Waveform export
    # ============================================================

    def _export_q1_data(
        self,
        sc_name: str,
        t: np.ndarray,
        v: np.ndarray,
        f_true: np.ndarray,
        scenario_meta: Dict[str, Any],
        m_name: str,
        f_hat: np.ndarray,
        tuning: Dict[str, Any],
        latency: int,
        exec_t: float,
        mets: Dict[str, Any],
        perturb_stats: Dict[str, float],
    ) -> None:
        sc_path = os.path.join(self.out_dir, "waveforms", sc_name)
        _ensure_dir(sc_path)

        meta_path = os.path.join(sc_path, "scenario_meta.json")
        if not os.path.exists(meta_path):
            try:
                save_json(meta_path, _safe_jsonify(scenario_meta))
            except Exception:
                save_json(meta_path, {"scenario_meta": str(scenario_meta)})

        pd.DataFrame({"t": t, "real_v": v, "real_f": f_true}).to_csv(
            os.path.join(sc_path, "ground_truth.csv"), index=False
        )
        pd.DataFrame({"t": t, "f_hat": f_hat}).to_csv(
            os.path.join(sc_path, f"{m_name}.csv"), index=False
        )
        save_json(
            os.path.join(sc_path, f"{m_name}_meta.json"),
            {
                "method": m_name,
                "tuning": _safe_jsonify(tuning),
                "latency_samples": int(latency),
                "effective_latency_s": float(latency)
                / max(float(self.cfg.fs_dsp_hz), 1.0),
                "exec_t": float(exec_t),
                "metrics_raw": _safe_jsonify(mets),
                "perturb_stats": _safe_jsonify(perturb_stats),
            },
        )
        _log(f"[WAVEFORMS] exported sc={sc_name} method={m_name} -> {sc_path}")


# ============================================================
# Minimal CLI (best-effort, project-dependent)
# ============================================================


def _try_load_cfg(path: str) -> ExperimentConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    for fn in ("from_yaml", "from_yml", "from_json", "load", "parse_file"):
        if hasattr(ExperimentConfig, fn):
            try:
                return getattr(ExperimentConfig, fn)(path)  # type: ignore[misc]
            except Exception:
                pass

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ExperimentConfig(**d)  # type: ignore[arg-type]
    if ext in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Need PyYAML to read {path}: {e}")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return ExperimentConfig(**d)  # type: ignore[arg-type]

    raise RuntimeError(
        f"Don't know how to load ExperimentConfig from {path}. "
        "Add ExperimentConfig.from_yaml()/load() or pass a .json/.yaml config."
    )


def _try_build_registry(cfg: ExperimentConfig) -> MethodRegistry:
    try:
        return MethodRegistry(cfg)  # type: ignore[call-arg]
    except Exception:
        try:
            return MethodRegistry()  # type: ignore[call-arg]
        except Exception as e:
            raise RuntimeError(
                f"Failed to build MethodRegistry: {type(e).__name__}: {e}"
            )


def _try_get_signals_catalog() -> Dict[str, Any]:
    try:
        import scenarios.ibg_events as ibg  # type: ignore

        for attr in ("SIGNALS", "signals", "SCENARIOS", "scenario_catalog", "CATALOG"):
            if hasattr(ibg, attr):
                x = getattr(ibg, attr)
                if isinstance(x, dict) and len(x) > 0:
                    return x
    except Exception:
        pass

    raise RuntimeError(
        "Could not find a scenario catalog dict (keys=scenario_ids). "
        "Expose one in scenarios/ibg_events.py (e.g., SIGNALS={...}) "
        "or call MonteCarloRunner.run(signals=<dict>) from your experiment script."
    )


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage:\n  python mc.py <path/to/experiment_cfg.(yaml|yml|json)>",
            file=sys.stderr,
        )
        return 2

    cfg_path = argv[1]
    cfg = _try_load_cfg(cfg_path)
    registry = _try_build_registry(cfg)
    runner = MonteCarloRunner(cfg, registry)

    signals = _try_get_signals_catalog()
    runner.run(signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
