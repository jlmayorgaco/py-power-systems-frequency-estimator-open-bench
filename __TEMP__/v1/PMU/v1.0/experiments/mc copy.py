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
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _fmt_params(d: Dict[str, Any]) -> str:
    if not d:
        return "default"
    items: List[str] = []
    for k in sorted(d.keys()):
        if k == "label":
            continue
        items.append(f"{k}={d[k]}")
    return ", ".join(items) if items else "default"


def _percentiles(x: np.ndarray, ps: List[int]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


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
        if "raw" in x:
            try:
                v = float(x["raw"])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
        if "value" in x and x["value"] is not None:
            try:
                v = float(x["value"])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
        for k in ("mean", "val", "score"):
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


def _safe_len(x: Any) -> int:
    try:
        return int(len(x))
    except Exception:
        return 0


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
        if isinstance(sc, dict) and "T" in sc:
            return float(sc["T"])
        return 5.0

    def _regen_scenario(self, scenario_id: str, seed: int):
        fs_phys = float(getattr(self.cfg, "fs_physics_hz", self.cfg.fs_dsp_hz))
        T = float(self._scenario_T())
        return get_test_signal(scenario_id=scenario_id, fs=fs_phys, T=T, seed=int(seed))

    def _infer_latency(self, est: Any) -> int:
        for attr in ("latency_samples", "window_samples", "sz", "maf_win"):
            if hasattr(est, attr):
                try:
                    return int(getattr(est, attr))
                except Exception:
                    pass
        return 0

    def _split_indices(self, n: int, tune_frac: float):
        cut = int(round(n * max(0.0, min(1.0, float(tune_frac)))))
        return np.arange(0, cut), np.arange(cut, n)

    def _perturb_voltage(
        self,
        v: np.ndarray,
        rng: np.random.Generator,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """
        Monte Carlo perturbations (Q1-clean):
        - Keep variations bounded (1%–5% max by default).
        - No huge impulsive spikes.
        - Preserve scenario physics: do NOT override the designed event waveform.

        Controls expected in cfg.mc.perturb (all optional):
            amp_pct_jitter: float  (std dev of multiplicative amp jitter, e.g. 0.01)
            amp_cap_pct: float     (hard cap on |amp jitter|, default 0.05)
            noise_rms_pct: float   (AWGN RMS as fraction of signal RMS, default 0.01)
            noise_rms_cap_pct: float (cap on AWGN RMS fraction, default 0.03)
            impulsive_prob: float  (probability per sample, e.g. 0.002)
            impulsive_scale_pct: float (std dev of spikes as fraction of signal RMS, default 0.01)
            impulsive_cap_pct: float   (hard cap on |spike| as fraction of signal RMS, default 0.05)
            allow_impulses_in: list[str] (scenario IDs allowed to get impulses; default ["G3_E13_Impulsive_Outliers"])
        """
        sc_id = str(meta.get("scenario_id", ""))

        mc = getattr(self.cfg, "mc", None)
        if not mc:
            return np.asarray(v, dtype=float).copy()

        pert = _cfg_get(mc, "perturb", None)
        if not pert:
            return np.asarray(v, dtype=float).copy()

        out = np.asarray(v, dtype=float).copy()
        n = out.size
        if n == 0:
            return out

        # --- robust signal scale (RMS) ---
        sig_rms = float(np.sqrt(np.mean(out * out) + 1e-12))

        # ------------------------------------------------------------------
        # 1) Multiplicative amplitude jitter (bounded)
        # ------------------------------------------------------------------
        amp_std = float(_cfg_get(pert, "amp_pct_jitter", 0.0))
        amp_cap = float(_cfg_get(pert, "amp_cap_pct", 0.05))  # 5% max by default

        if amp_std > 0:
            # one global gain per realization (not per-sample)
            g = float(1.0 + rng.normal(0.0, amp_std))
            g = float(np.clip(g, 1.0 - amp_cap, 1.0 + amp_cap))
            out *= g

        # ------------------------------------------------------------------
        # 2) Additive Gaussian noise (bounded RMS fraction)
        # ------------------------------------------------------------------
        # IMPORTANT: do not add extra AWGN to "Pure" baseline unless explicitly desired
        if "Pure" not in sc_id:
            noise_rms_pct = float(_cfg_get(pert, "noise_rms_pct", 0.0))
            noise_rms_cap = float(_cfg_get(pert, "noise_rms_cap_pct", 0.03))  # 3% cap

            if noise_rms_pct > 0:
                nr = float(np.clip(noise_rms_pct, 0.0, noise_rms_cap))
                noise_std = nr * sig_rms
                out += rng.normal(0.0, noise_std, size=out.shape)

        # ------------------------------------------------------------------
        # 3) Impulsive micro-spikes (bounded amplitude)
        # ------------------------------------------------------------------
        # Default: impulses ONLY in the dedicated outlier scenario.
        allow_impulses_in = _cfg_get(pert, "allow_impulses_in", None)
        if not isinstance(allow_impulses_in, (list, tuple)):
            allow_impulses_in = ["G3_E13_Impulsive_Outliers"]

        # If you want Composite Islanding clean, keep it out of allow_impulses_in.
        if sc_id in set(map(str, allow_impulses_in)):
            p_imp = float(_cfg_get(pert, "impulsive_prob", 0.0))
            scale_pct = float(_cfg_get(pert, "impulsive_scale_pct", 0.01))  # 1% RMS
            cap_pct = float(_cfg_get(pert, "impulsive_cap_pct", 0.05))  # 5% RMS cap

            if p_imp > 0 and scale_pct > 0:
                mask = rng.random(size=out.shape) < p_imp
                k = int(np.sum(mask))
                if k > 0:
                    spikes = rng.normal(0.0, scale_pct * sig_rms, size=k)
                    # hard bound spikes to avoid "vertical lines"
                    lim = cap_pct * sig_rms
                    spikes = np.clip(spikes, -lim, +lim)
                    out[mask] += spikes

        # Final hard safety: never let perturbation exceed a reasonable bound vs RMS
        # (prevents pathological configs)
        hard_cap = float(_cfg_get(pert, "hard_clip_rms_mult", 6.0))  # e.g., ±6*rms
        out = np.clip(out, -hard_cap * sig_rms, +hard_cap * sig_rms)

        return out

    def run(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        mc = getattr(self.cfg, "mc", None)
        n_train, n_test = int(mc["n_train_seeds"]), int(mc["n_test_seeds"])
        base_seed, tune_frac = int(mc["base_seed"]), float(mc["tune_frac"])
        scenario_names = self._scenario_names_from_signals(signals)
        methods = list(getattr(self.cfg, "methods", []))

        best_params_per_method = {m: {} for m in methods}
        acc = {sc: {m: {} for m in methods} for sc in scenario_names}
        pbar = tqdm(
            total=(n_train + n_test) * len(scenario_names) * len(methods),
            desc="MC Full Range",
            unit="run",
        )

        try:
            phases = [("train", n_train, 0), ("test", n_test, n_train)]
            for phase, n_seeds, phase_seed_offset in phases:
                for seed_idx in range(n_seeds):
                    global_seed = base_seed + phase_seed_offset + seed_idx
                    rng = np.random.default_rng(global_seed)

                    for sc_name in scenario_names:
                        t_phys, v_ana, f_true, meta = self._regen_scenario(
                            sc_name, seed=global_seed
                        )
                        ratio = int(self.cfg.downsampling_ratio)
                        v_ds, f_ds, t_ds = (
                            v_ana[::ratio],
                            f_true[::ratio],
                            t_phys[::ratio],
                        )
                        v_noisy = self._perturb_voltage(v_ds, rng, meta)

                        # --- ESTRATEGIA FULL RANGE (0s a 5s) ---
                        # Sintonizamos en un fragmento pero evaluamos la señal COMPLETA
                        idx_fit, _ = self._split_indices(len(v_ds), tune_frac)
                        idx_eval = np.arange(len(v_ds))

                        v_fit, f_fit = v_noisy[idx_fit], f_ds[idx_fit]
                        v_eval, f_eval, t_eval = (
                            v_noisy[idx_eval],
                            f_ds[idx_eval],
                            t_ds[idx_eval],
                        )
                        is_export_seed = phase == "test" and seed_idx == 0

                        for m in methods:
                            spec = self.registry.get(m)
                            make_est = spec.builder
                            base_params = {"fs_hz": float(self.cfg.fs_dsp_hz)}

                            if phase == "train" and (m in self.grids):

                                def rmse_score(trace):
                                    L = min(len(trace), len(f_fit))
                                    return (
                                        np.sqrt(np.mean((trace[:L] - f_fit[:L]) ** 2))
                                        if L > 0
                                        else 1e6
                                    )

                                res = tune_generic(
                                    m,
                                    self.grids[m],
                                    make_est,
                                    rmse_score,
                                    v_fit,
                                    base_params=base_params,
                                )
                                best_params_per_method[m] = dict(res.params)

                            est = make_est({**base_params, **best_params_per_method[m]})
                            if hasattr(est, "reset"):
                                est.reset()

                            # Ejecución en tiempo real causal de 0 a 5 segundos
                            t_start = time.perf_counter()
                            f_hat = np.array(
                                [est.step(float(x)) for x in v_eval], dtype=float
                            )
                            exec_t = float(time.perf_counter() - t_start)

                            latency = self._infer_latency(est)
                            mets = compute_metrics(
                                f_hat, f_eval, exec_t, latency, self.metric_cfg, sc_name
                            )

                            if is_export_seed:
                                self._export_q1_data(
                                    sc_name=sc_name,
                                    t=t_eval,
                                    v=v_eval,
                                    f_true=f_eval,
                                    m_name=m,
                                    f_hat=f_hat,
                                    tuning=best_params_per_method[m],
                                    latency=latency,
                                    exec_t=exec_t,
                                    mets=mets,
                                )

                            if phase == "test":
                                for k, val in mets.items():
                                    num = _metric_obj_to_float(val)
                                    if num is not None:
                                        acc[sc_name][m].setdefault(k, []).append(num)
                            pbar.update(1)

            # Guardado masivo de resultados estadísticos
            final_res = {"metadata": mc, "results": {}}
            for sc in scenario_names:
                final_res["results"][sc] = {"methods": {}}
                for m in methods:
                    m_mets = {}
                    for k, vals in acc[sc][m].items():
                        fin = np.array(vals)[np.isfinite(vals)]
                        if fin.size:
                            m_mets[k] = {
                                "mean": float(np.mean(fin)),
                                "std": float(np.std(fin)),
                                **_percentiles(fin, [5, 50, 95]),
                            }
                    final_res["results"][sc]["methods"][m] = m_mets

            save_json(f"{self.out_dir}/mc_results.json", final_res)
            return final_res
        finally:
            pbar.close()

    def _export_q1_data(
        self, sc_name, t, v, f_true, m_name, f_hat, tuning, latency, exec_t, mets
    ):
        """Persistencia centralizada en artifacts/."""
        sc_path = os.path.join(self.out_dir, "waveforms", sc_name)
        os.makedirs(sc_path, exist_ok=True)
        # Sincronización completa de los 5 segundos de datos
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
                "tuning": tuning,
                "latency": int(latency),
                "exec_t": exec_t,
                "metrics": mets,
            },
        )
