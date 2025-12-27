from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.config import ExperimentConfig
from experiments.registry import MethodRegistry
from experiments.tuning import build_grids, tune_generic
from experiments.io import build_run_record, save_json

from domain.metrics_api import compute_metrics
from domain.metrics_base import MetricConfig

# True Monte Carlo: regenerate scenarios per seed
from scenarios.ibg_events import get_test_signal

# ============================================================
# Helpers
# ============================================================

def _cfg_get(obj: Any, key: str, default: Any) -> Any:
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def _fmt_params(d: Dict[str, Any]) -> str:
    if not d: return "default"
    items: List[str] = []
    for k in sorted(d.keys()):
        if k == "label": continue
        items.append(f"{k}={d[k]}")
    return ", ".join(items) if items else "default"

def _percentiles(x: np.ndarray, ps: List[int]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0: return {f"p{p}": float("nan") for p in ps}
    out: Dict[str, float] = {}
    for p in ps: out[f"p{p}"] = float(np.percentile(x, p))
    return out

def _metric_obj_to_float(x: Any) -> Optional[float]:
    if isinstance(x, bool): return None
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
        except Exception: return None
    if isinstance(x, dict):
        if "raw" in x:
            try:
                v = float(x["raw"])
                if np.isfinite(v): return v
            except Exception: pass
        if "value" in x and x["value"] is not None:
            try:
                v = float(x["value"])
                if np.isfinite(v): return v
            except Exception: pass
        for k in ("mean", "val", "score"):
            if k in x: return _metric_obj_to_float(x[k])
    for attr in ("raw", "value", "mean", "score"):
        if hasattr(x, attr):
            try: return _metric_obj_to_float(getattr(x, attr))
            except Exception: pass
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception: return None

def _safe_len(x: Any) -> int:
    try: return int(len(x))
    except Exception: return 0

# ============================================================
# Runner
# ============================================================

class MonteCarloRunner:
    def __init__(self, cfg: ExperimentConfig, registry: MethodRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.grids = build_grids(getattr(cfg, "grids", {}))
        self.metric_cfg = MetricConfig(fs_hz=float(self.cfg.fs_dsp_hz))
        # Centralización obligatoria en artifacts
        self.out_dir = "artifacts/results_mc"
        os.makedirs(self.out_dir, exist_ok=True)

    def _scenario_names_from_signals(self, signals) -> List[str]:
        return sorted(list(signals.keys()))

    def _scenario_T(self) -> float:
        sc = getattr(self.cfg, "scenario", None)
        if isinstance(sc, dict) and "T" in sc: return float(sc["T"])
        return 5.0

    def _regen_scenario(self, scenario_id: str, seed: int):
        fs_phys = float(getattr(self.cfg, "fs_physics_hz", self.cfg.fs_dsp_hz))
        T = float(self._scenario_T())
        return get_test_signal(scenario_id=scenario_id, fs=fs_phys, T=T, seed=int(seed))

    def _infer_latency(self, est: Any) -> int:
        for attr in ("latency_samples", "window_samples", "sz", "maf_win"):
            if hasattr(est, attr):
                try: return int(getattr(est, attr))
                except Exception: pass
        return 0

    def _split_indices(self, n: int, tune_frac: float):
        cut = int(round(n * max(0.0, min(1.0, float(tune_frac)))))
        return np.arange(0, cut), np.arange(cut, n)

    def _perturb_voltage(self, v: np.ndarray, rng: np.random.Generator, meta: Dict[str, Any]) -> np.ndarray:
        """Filtro de pureza Q1: Evita ruidos impulsivos en baselines."""
        sc_id = meta.get("scenario_id", "")
        mc = getattr(self.cfg, "mc", None)
        if not mc: return v.copy()
        pert = _cfg_get(mc, "perturb", None)
        if not pert: return v.copy()
        out = np.asarray(v, dtype=float).copy()
        
        amp_j = float(_cfg_get(pert, "amp_pct_jitter", 0.0))
        if amp_j > 0: out *= (1.0 + rng.normal(0.0, amp_j))
        
        if "Pure" not in sc_id:
            snr_j = float(_cfg_get(pert, "snr_db_jitter", 0.0))
            if snr_j > 0:
                snr0 = float(meta.get("snr_db", 40.0))
                snr = snr0 + rng.normal(0.0, snr_j)
                noise_std = np.sqrt((np.mean(out**2)+1e-12) / (10.0**(snr/10.0)))
                out += rng.normal(0.0, noise_std, size=out.shape)
            
        # Bloqueo selectivo de ruido impulsivo para proteger la Physics Layer
        if "G1" not in sc_id and "G2" not in sc_id:
            p_imp = float(_cfg_get(pert, "impulsive_prob", 0.0))
            scale_imp = float(_cfg_get(pert, "impulsive_scale", 0.0))
            if p_imp > 0 and scale_imp > 0:
                mask = rng.random(size=out.shape) < p_imp
                out[mask] += rng.normal(0.0, scale_imp * (np.std(out)+1e-12), size=int(np.sum(mask)))
        return out

    def run(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        mc = getattr(self.cfg, "mc", None)
        n_train, n_test = int(mc["n_train_seeds"]), int(mc["n_test_seeds"])
        base_seed, tune_frac = int(mc["base_seed"]), float(mc["tune_frac"])
        scenario_names = self._scenario_names_from_signals(signals)
        methods = list(getattr(self.cfg, "methods", []))
        
        best_params_per_method = {m: {} for m in methods}
        acc = {sc: {m: {} for m in methods} for sc in scenario_names}
        pbar = tqdm(total=(n_train + n_test) * len(scenario_names) * len(methods), desc="MC Full Range", unit="run")

        try:
            phases = [("train", n_train, 0), ("test", n_test, n_train)]
            for phase, n_seeds, phase_seed_offset in phases:
                for seed_idx in range(n_seeds):
                    global_seed = base_seed + phase_seed_offset + seed_idx
                    rng = np.random.default_rng(global_seed)

                    for sc_name in scenario_names:
                        t_phys, v_ana, f_true, meta = self._regen_scenario(sc_name, seed=global_seed)
                        ratio = int(self.cfg.downsampling_ratio)
                        v_ds, f_ds, t_ds = v_ana[::ratio], f_true[::ratio], t_phys[::ratio]
                        v_noisy = self._perturb_voltage(v_ds, rng, meta)
                        
                        # --- ESTRATEGIA FULL RANGE (0s a 5s) ---
                        # Sintonizamos en un fragmento pero evaluamos la señal COMPLETA
                        idx_fit, _ = self._split_indices(len(v_ds), tune_frac)
                        idx_eval = np.arange(len(v_ds)) 
                        
                        v_fit, f_fit = v_noisy[idx_fit], f_ds[idx_fit]
                        v_eval, f_eval, t_eval = v_noisy[idx_eval], f_ds[idx_eval], t_ds[idx_eval]
                        is_export_seed = (phase == "test" and seed_idx == 0)

                        for m in methods:
                            spec = self.registry.get(m)
                            make_est = spec.builder
                            base_params = {"fs_hz": float(self.cfg.fs_dsp_hz)}

                            if phase == "train" and (m in self.grids):
                                def rmse_score(trace):
                                    L = min(len(trace), len(f_fit))
                                    return np.sqrt(np.mean((trace[:L]-f_fit[:L])**2)) if L>0 else 1e6
                                res = tune_generic(m, self.grids[m], make_est, rmse_score, v_fit, base_params=base_params)
                                best_params_per_method[m] = dict(res.params)

                            est = make_est({**base_params, **best_params_per_method[m]})
                            if hasattr(est, "reset"): est.reset()

                            # Ejecución en tiempo real causal de 0 a 5 segundos
                            t_start = time.perf_counter()
                            f_hat = np.array([est.step(float(x)) for x in v_eval], dtype=float)
                            exec_t = float(time.perf_counter() - t_start)
                            
                            latency = self._infer_latency(est)
                            mets = compute_metrics(f_hat, f_eval, exec_t, latency, self.metric_cfg, sc_name)

                            if is_export_seed:
                                self._export_q1_data(
                                    sc_name=sc_name, t=t_eval, v=v_eval, f_true=f_eval,
                                    m_name=m, f_hat=f_hat, tuning=best_params_per_method[m],
                                    latency=latency, exec_t=exec_t, mets=mets
                                )

                            if phase == "test":
                                for k, val in mets.items():
                                    num = _metric_obj_to_float(val)
                                    if num is not None: acc[sc_name][m].setdefault(k, []).append(num)
                            pbar.update(1)

            # Guardado masivo de resultados estadísticos
            final_res = {"metadata": mc, "results": {}}
            for sc in scenario_names:
                final_res["results"][sc] = {"methods": {}}
                for m in methods:
                    m_mets = {}
                    for k, vals in acc[sc][m].items():
                        fin = np.array(vals)[np.isfinite(vals)]
                        if fin.size: m_mets[k] = {"mean": float(np.mean(fin)), "std": float(np.std(fin)), **_percentiles(fin, [5, 50, 95])}
                    final_res["results"][sc]["methods"][m] = m_mets
            
            save_json(f"{self.out_dir}/mc_results.json", final_res)
            return final_res
        finally: pbar.close()

    def _export_q1_data(self, sc_name, t, v, f_true, m_name, f_hat, tuning, latency, exec_t, mets):
        """Persistencia centralizada en artifacts/."""
        sc_path = os.path.join(self.out_dir, "waveforms", sc_name)
        os.makedirs(sc_path, exist_ok=True)
        # Sincronización completa de los 5 segundos de datos
        pd.DataFrame({'t': t, 'real_v': v, 'real_f': f_true}).to_csv(os.path.join(sc_path, "ground_truth.csv"), index=False)
        pd.DataFrame({'t': t, 'f_hat': f_hat}).to_csv(os.path.join(sc_path, f"{m_name}.csv"), index=False)
        save_json(os.path.join(sc_path, f"{m_name}_meta.json"), {
            "method": m_name, "tuning": tuning,
            "latency": int(latency), "exec_t": exec_t, "metrics": mets
        })