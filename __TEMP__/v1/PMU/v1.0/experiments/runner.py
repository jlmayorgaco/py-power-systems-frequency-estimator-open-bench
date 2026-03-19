from __future__ import annotations
import os
import time
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
from typing import Dict, Any, Tuple, List

from experiments.config import ExperimentConfig
from experiments.registry import MethodRegistry
from experiments.tuning import build_grids, tune_generic
from experiments.io import save_json

from domain.metrics_api import compute_metrics
from domain.metrics_base import MetricConfig


class BenchmarkRunner:
    def __init__(self, cfg: ExperimentConfig, registry: MethodRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.grids = build_grids(cfg.grids)
        self.fs_dsp = float(self.cfg.fs_dsp_hz)
        self.metric_cfg = MetricConfig(fs_hz=self.fs_dsp)
        self.output_dir = "results_q1_final"
        os.makedirs(self.output_dir, exist_ok=True)

    def _downsample_q1(self, v_ana: np.ndarray, ratio: int) -> np.ndarray:
        """Filtro Antialiasing FIR (AAF) antes del diezmado físico."""
        if ratio <= 1:
            return v_ana
        return signal.decimate(v_ana, ratio, ftype="fir")

    def _apply_physics_noise(
        self, v: np.ndarray, snr_db: float, seed: int
    ) -> np.ndarray:
        """Inyección de ruido en la capa analógica (1 MHz o fs_physics)."""
        rng = np.random.default_rng(seed)
        sig_pwr = np.mean(v**2)
        noise_pwr = sig_pwr / (10 ** (snr_db / 10))
        return v + rng.normal(0, np.sqrt(noise_pwr), size=len(v))

    def _run_method_profiled(
        self, spec, tuning: Dict[str, Any], v: np.ndarray
    ) -> Tuple[np.ndarray, float, int]:
        """Mide tiempo de ejecución y extrae latencia algorítmica."""
        builder = getattr(spec, "factory", getattr(spec, "builder", None))
        est = builder(tuning)
        if hasattr(est, "reset"):
            est.reset()

        t_start = time.perf_counter()
        out = np.array([est._step(float(x)) for x in v], dtype=float)
        t_end = time.perf_counter()

        total_time_s = t_end - t_start
        lat = int(getattr(est, "latency_samples", 0))
        return out, total_time_s, lat

    def run(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Definiciones de métricas para que el JSON sea auto-explicativo (Sugerencia Q1)
        export = {
            "metadata": {
                **self.cfg.__dict__,
                "metrics_definitions": {
                    "RMSE": {
                        "unit": "Hz",
                        "desc": "Root Mean Square Error (Excluyendo Warm-up)",
                    },
                    "MAE": {"unit": "Hz", "desc": "Mean Absolute Error"},
                    "RFE_RMSE": {
                        "unit": "Hz/s^2",
                        "desc": "RoCoF Frequency Error (Filtered)",
                    },
                    "TRIP_TIME": {
                        "unit": "s",
                        "desc": "Latencia de detección desde t=0",
                    },
                    "TIME_PER_SAMPLE_US": {
                        "unit": "us",
                        "desc": "Costo de CPU por ejecución efectiva",
                    },
                    "CVAR95": {
                        "unit": "Hz",
                        "desc": "Riesgo en el peor 5% de los casos",
                    },
                },
                "execution_info": {
                    "node": os.uname().nodename if hasattr(os, "uname") else "unknown",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            "results": {},
        }

        scenario_names = sorted(signals.keys())
        pbar = tqdm(
            total=len(scenario_names) * len(self.cfg.methods), desc="Q1 Monte Carlo"
        )

        for sc_name in scenario_names:
            t_phys, v_ana_pure, f_true_phys, meta = signals[sc_name]
            ratio = int(self.cfg.downsampling_ratio)
            f_true_dsp = f_true_phys[::ratio]

            # Trazabilidad de semillas (Audit trail)
            mc_seeds = [self.cfg.seed + k for k in range(self.cfg.mc.n_test_seeds)]

            export["results"][sc_name] = {
                "scenario_meta": {
                    **meta,
                    "mc_seeds": mc_seeds,
                    "fs_effective_hz": self.fs_dsp,
                },
                "methods": {},
            }

            snr_db = self.cfg.mc.perturb.get("snr_db_jitter", 40.0)

            for m_name in self.cfg.methods:
                spec = self.registry.get(m_name)

                # --- 2. ORACLE TUNING ---
                v_clean_dsp = self._downsample_q1(v_ana_pure, ratio)

                def score_fn(trace):
                    # Solo sintonizamos sobre la parte estable (excluyendo transitorio inicial)
                    warm = int(0.2 * self.fs_dsp)
                    msk = np.isfinite(trace[warm:]) & np.isfinite(f_true_dsp[warm:])
                    if not np.any(msk):
                        return 1e9
                    return np.sqrt(
                        np.mean((trace[warm:][msk] - f_true_dsp[warm:][msk]) ** 2)
                    )

                res_tune = tune_generic(
                    m_name, self.grids[m_name], spec.factory, score_fn, v_clean_dsp
                )
                best_tuning = res_tune.params
                decim = best_tuning.get("decim", 1)

                # --- 3. MONTE CARLO LOOP ---
                mc_iteration_results = []
                for k in mc_seeds:
                    # Inyección física (AAF real)
                    v_noisy_phys = self._apply_physics_noise(v_ana_pure, snr_db, seed=k)
                    v_input_dsp = self._downsample_q1(v_noisy_phys, ratio)

                    f_raw, total_time_s, lat = self._run_method_profiled(
                        spec, best_tuning, v_input_dsp
                    )

                    # Honestidad Computacional: escalamos el tiempo por el factor de diezmado
                    # para que metrics_api reporte el tiempo real por ejecución de bloque.
                    time_to_pass = total_time_s * decim

                    # Alineación temporal para RMSE (Accuracy)
                    f_aligned = np.full_like(f_raw, np.nan)
                    if lat < len(f_raw):
                        f_aligned[:-lat] = f_raw[lat:]

                    # Cálculo de métricas ricas
                    # m_prec: RMSE/MAE/RFE sobre señal alineada
                    # m_speed: TRIP_TIME sobre señal RAW (causalidad)
                    m_prec = compute_metrics(
                        f_aligned,
                        f_true_dsp,
                        time_to_pass,
                        lat,
                        self.metric_cfg,
                        sc_name,
                    )
                    m_speed = compute_metrics(
                        f_raw, f_true_dsp, time_to_pass, 0, self.metric_cfg, sc_name
                    )

                    combined = {**m_prec}
                    # Inyectamos métricas de protección de la ejecución RAW
                    for k_trip in m_speed.keys():
                        if "TRIP_TIME" in k_trip or "SETTLING" in k_trip:
                            combined[k_trip] = m_speed[k_trip]

                    mc_iteration_results.append(combined)

                # 4. AGREGACIÓN Y METADATOS DEL MÉTODO
                method_results = self._aggregate_mc_metrics(mc_iteration_results)
                method_results["method_info"] = {
                    "optimal_params": best_tuning,
                    "window_samples": int(
                        best_tuning.get("window_cycles", 0) * (self.fs_dsp / 60)
                    ),
                    "latency_s": lat / self.fs_dsp,
                    "effective_fs_hz": self.fs_dsp / decim,
                }

                export["results"][sc_name]["methods"][m_name] = method_results
                pbar.update(1)

        save_json(f"{self.output_dir}/benchmark_results_q1.json", export)
        return export

    def _aggregate_mc_metrics(self, mc_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregación estadística completa para Journal."""
        summary = {}
        if not mc_results:
            return summary

        # Identificar todas las métricas presentes
        metric_keys = mc_results[0].keys()

        for key in metric_keys:
            # Extraer valores raw para promediar
            vals = [
                r[key].get("value") if isinstance(r[key], dict) else r[key]
                for r in mc_results
            ]
            vals = [v for v in vals if v is not None and np.isfinite(v)]

            if not vals:
                summary[key] = {"mean": None, "status": "all_failed"}
                continue

            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "n_finite": len(vals),
            }
        return summary
