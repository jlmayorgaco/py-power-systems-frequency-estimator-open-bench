#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kundur Two-Area (4-machine) — Hidden-Margin Q1 Demo (ANDES-backed)
=================================================================

Runs ANDES Kundur test case, extracts rotor angle/speed time series,
builds a robust generator-to-generator coupling proxy B_ij using the
*shortest-path reactance* between generator buses:

  Xsp(i,j) = shortest path sum of |x| over the bus graph
  B_ij     = 1 / Xsp(i,j)   (connected case => dense, nonzero)

Then computes operating-point geometry indicators:

  W_eff = B_ij * cos(delta_i* - delta_j*)
  H_eff = M^{-1/2} L(W_eff) M^{-1/2}
  lambda2(H_eff), Fiedler energy, Forman-Ricci curvature bottlenecks

Saves:
  results_kundur2_hidden_margin_q1_andes/figs/*.png and *.pdf
  results_kundur2_hidden_margin_q1_andes/diagnostic.json

Dependencies:
  pip install numpy networkx matplotlib andes

Run:
  python kundur_hidden_margin_q1_andes.py

Optional:
  ANDES_CASE=kundur/kundur_full.xlsx python kundur_hidden_margin_q1_andes.py
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# IEEE-ish plotting defaults
# ============================================================
def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


# ============================================================
# I/O
# ============================================================
class IO:
    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(path: str, obj: Dict[str, Any]) -> None:
        def _convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            return o

        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_convert)


# ============================================================
# Configs
# ============================================================
@dataclass(frozen=True)
class Paths:
    base_dir: str
    results_dir: str
    figs_dir: str

    @staticmethod
    def from_script(__file__: str, results_folder: str) -> "Paths":
        base = os.path.dirname(os.path.abspath(__file__))
        results = os.path.join(base, results_folder)
        figs = os.path.join(results, "figs")
        return Paths(base_dir=base, results_dir=results, figs_dir=figs)


@dataclass(frozen=True)
class GraphConfig:
    # kept only for metadata (not used to build ANDES plant)
    w_intra: float = 8.0
    w_tie: float = 0.6


@dataclass(frozen=True)
class EventConfig:
    # TDS horizon
    T: float = 20.0
    # used only as fallback for dt
    dt_guess: float = 1.0 / 30.0

    # windows (seconds)
    win_pre: Tuple[float, float] = (0.5, 3.0)
    win_post: Tuple[float, float] = (4.0, 8.0)
    win_precrisis: Tuple[float, float] = (8.5, 12.0)
    win_crisis: Tuple[float, float] = (12.0, 18.0)

    # warn thresholds
    lam2_eff_warn: float = 0.25
    min_cos_warn: float = 0.20

    # crisis detection thresholds
    # NOTE: omega thresholds are applied to *deviation* from pre-window reference
    delta_sep_trip_rad: float = 1.45
    omega_dev_trip: float = 0.01  # pu deviation (typical). override if you want.
    rocof_trip: float = 0.25  # pu/s (on omega deviation)
    crisis_sustain_sec: float = 0.5


# ============================================================
# Graph / matrices
# ============================================================
class Lap:
    @staticmethod
    def laplacian_from_weights(W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        return np.diag(np.sum(W, axis=1)) - W


# ============================================================
# Spectral + geometry
# ============================================================
class Spectral:
    @staticmethod
    def inertial_laplacian(L: np.ndarray, M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(1e-12, M)))
        return inv_sqrt @ L @ inv_sqrt

    @staticmethod
    def lambda2(H: np.ndarray, tol: float = 1e-10) -> float:
        w = np.sort(np.linalg.eigvalsh(H))
        for val in w:
            if val > tol:
                return float(val)
        return 0.0

    @staticmethod
    def fiedler_vector(H: np.ndarray, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        w, V = np.linalg.eigh(H)
        order = np.argsort(w)
        w = w[order]
        V = V[:, order]
        for k in range(len(w)):
            if w[k] > tol:
                return float(w[k]), V[:, k]
        return 0.0, np.zeros(H.shape[0], dtype=float)


class Curvature:
    @staticmethod
    def forman_ricci_edges_from_W(
        labels: List[str], W: np.ndarray
    ) -> Dict[Tuple[str, str], float]:
        n = len(labels)
        nbrs = {i: [j for j in range(n) if W[i, j] > 0] for i in range(n)}
        F: Dict[Tuple[str, str], float] = {}

        for i in range(n):
            for j in nbrs[i]:
                if j <= i:
                    continue
                w_e = float(W[i, j])

                sum_i = 0.0
                for k in nbrs[i]:
                    if k == j:
                        continue
                    sum_i += 1.0 / math.sqrt(max(1e-12, w_e * float(W[i, k])))

                sum_j = 0.0
                for k in nbrs[j]:
                    if k == i:
                        continue
                    sum_j += 1.0 / math.sqrt(max(1e-12, w_e * float(W[j, k])))

                F_e = 2.0 - w_e * (sum_i + sum_j)
                u, v = labels[i], labels[j]
                key = (u, v) if u < v else (v, u)
                F[key] = float(F_e)
        return F

    @staticmethod
    def node_fragility_min_incident(
        labels: List[str], W: np.ndarray, F: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        n = len(labels)
        out: Dict[str, float] = {}
        for i in range(n):
            vals = []
            for j in range(n):
                if i == j or W[i, j] <= 0:
                    continue
                u, v = labels[i], labels[j]
                key = (u, v) if u < v else (v, u)
                if key in F:
                    vals.append(F[key])
            out[labels[i]] = float(np.min(vals)) if vals else float("nan")
        return out


class OperatingPointGeometry:
    @staticmethod
    def effective_weights(
        B: np.ndarray, delta_star: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        B = np.asarray(B, dtype=float)
        n = B.shape[0]
        W_eff = np.zeros((n, n), dtype=float)
        cos_mat = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                if B[i, j] <= 0:
                    continue
                d = float(delta_star[i] - delta_star[j])
                c = math.cos(d)
                cos_mat[i, j] = cos_mat[j, i] = c
                w = float(B[i, j] * c)
                # keep only positive effective couplings (standard swing linearization region)
                if w > 0.0:
                    W_eff[i, j] = W_eff[j, i] = w
        return W_eff, cos_mat

    @staticmethod
    def summarize(
        labels: List[str], B: np.ndarray, M: np.ndarray, delta_star: np.ndarray
    ) -> Dict[str, Any]:
        W_eff, cos_mat = OperatingPointGeometry.effective_weights(B, delta_star)
        L_eff = Lap.laplacian_from_weights(W_eff)
        H_eff = Spectral.inertial_laplacian(L_eff, M)
        lam2_eff = Spectral.lambda2(H_eff)

        eig, fied = Spectral.fiedler_vector(H_eff)
        denom = max(1e-12, float(np.sum(fied**2)))
        node_energy = (fied**2) / denom

        F = Curvature.forman_ricci_edges_from_W(labels, W_eff)
        F_vals = np.array(list(F.values()), dtype=float) if F else np.array([])
        F_min = float(np.min(F_vals)) if F_vals.size else float("nan")
        F_mean = float(np.mean(F_vals)) if F_vals.size else float("nan")
        bottlenecks = sorted(F.items(), key=lambda kv: kv[1])[:2]
        node_frag = Curvature.node_fragility_min_incident(labels, W_eff, F)

        # min cos over all (proxy) couplings in B
        cos_edges = []
        n = B.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if B[i, j] > 0:
                    cos_edges.append(float(cos_mat[i, j]))
        min_cos = float(np.min(cos_edges)) if cos_edges else float("nan")

        return {
            "delta_star_rad": delta_star,
            "min_cos_on_edges": min_cos,
            "lambda2_H_eff": float(lam2_eff),
            "critical_mode_eff": {
                "eigval": float(eig),
                "node_energy": node_energy,
            },
            "forman_ricci_eff": {
                "edge_curvature_min": F_min,
                "edge_curvature_mean": F_mean,
                "bottleneck_edges": [
                    {"edge": [u, v], "curv": float(c)} for (u, v), c in bottlenecks
                ],
                "node_fragility_min_incident_curv": node_frag,
            },
            "effective_weights_W_eff": W_eff,
            "effective_laplacian_L_eff": L_eff,
        }


# ============================================================
# Windows + compact indicators
# ============================================================
class Windows:
    @staticmethod
    def mask(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
        return (t >= t0) & (t <= t1)

    @staticmethod
    def slice(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return x[m, :]


class SOTAIndicators:
    @staticmethod
    def rocof(x: np.ndarray, dt: float) -> np.ndarray:
        if x.shape[0] < 2:
            return np.zeros((0, x.shape[1]))
        return np.diff(x, axis=0) / max(1e-12, dt)

    @staticmethod
    def summarize(
        t: np.ndarray,
        delta: np.ndarray,
        omega: np.ndarray,
        dt: float,
        t0: float,
        t1: float,
    ) -> Dict[str, Any]:
        m = Windows.mask(t, t0, t1)
        om = Windows.slice(omega, m)
        dd = Windows.slice(delta, m)

        ro = SOTAIndicators.rocof(om, dt=dt)
        abs_ro = np.abs(ro)

        mean_om_t = om.mean(axis=1) if om.size else np.array([])
        std_om_t = om.std(axis=1) if om.size else np.array([])

        # node-wise variance within the window (over time)
        var_nodes = np.var(om, axis=0) if om.size else np.zeros(omega.shape[1])

        nadir = float(np.min(om)) if om.size else float("nan")
        peak = float(np.max(om)) if om.size else float("nan")

        ro_p95 = float(np.percentile(abs_ro, 95)) if abs_ro.size else float("nan")
        ro_max = float(np.max(abs_ro)) if abs_ro.size else float("nan")

        if dd.size:
            max_sep_t = np.max(dd, axis=1) - np.min(dd, axis=1)
            max_sep = float(np.max(max_sep_t))
        else:
            max_sep = float("nan")

        return {
            "window": [float(t0), float(t1)],
            "omega": {
                "mean_over_time_mean": (
                    float(np.mean(mean_om_t)) if mean_om_t.size else float("nan")
                ),
                "mean_over_time_std": (
                    float(np.std(mean_om_t)) if mean_om_t.size else float("nan")
                ),
                "std_over_time_mean": (
                    float(np.mean(std_om_t)) if std_om_t.size else float("nan")
                ),
                "nadir": nadir,
                "peak": peak,
                "var_nodes": var_nodes,
            },
            "rocof": {
                "abs_p95": ro_p95,
                "abs_max": ro_max,
            },
            "angle_separation": {"max_delta_sep_rad": max_sep},
        }


# ============================================================
# Crisis detection (uses omega deviation)
# ============================================================
class CrisisDetector:
    @staticmethod
    def detect(
        t: np.ndarray,
        delta: np.ndarray,
        omega: np.ndarray,
        dt: float,
        omega_ref: np.ndarray,
        delta_sep_trip_rad: float,
        omega_dev_trip: float,
        rocof_trip: float,
        sustain_sec: float,
    ) -> Dict[str, Any]:
        sustain_n = max(1, int(sustain_sec / max(1e-12, dt)))

        # angle separation proxy
        max_sep = np.max(delta, axis=1) - np.min(delta, axis=1)
        ex_sep = max_sep >= delta_sep_trip_rad

        # omega deviation + rocof on deviation
        omega_dev = omega - omega_ref.reshape(1, -1)
        ex_om = np.any(np.abs(omega_dev) >= omega_dev_trip, axis=1)

        ro = np.diff(omega_dev, axis=0) / max(1e-12, dt)
        t_ro = t[1:]
        ex_ro = np.any(np.abs(ro) >= rocof_trip, axis=1)

        def first_sustained(ex_mask: np.ndarray, tt: np.ndarray) -> Optional[float]:
            run = 0
            for k in range(len(ex_mask)):
                if bool(ex_mask[k]):
                    run += 1
                    if run >= sustain_n:
                        return float(tt[k])
                else:
                    run = 0
            return None

        t_sep = first_sustained(ex_sep, t)
        t_om = first_sustained(ex_om, t)
        t_ro2 = first_sustained(ex_ro, t_ro)

        crisis = (t_sep is not None) or (t_om is not None) or (t_ro2 is not None)
        return {
            "crisis_detected": bool(crisis),
            "first_time_sep_trip": t_sep,
            "first_time_omega_dev_trip": t_om,
            "first_time_rocof_trip": t_ro2,
            "thresholds": {
                "delta_sep_trip_rad": float(delta_sep_trip_rad),
                "omega_dev_trip": float(omega_dev_trip),
                "rocof_trip": float(rocof_trip),
                "sustain_sec": float(sustain_sec),
            },
        }


# ============================================================
# ANDES adapter (Kundur case)
# ============================================================
@dataclass(frozen=True)
class AndesSimOut:
    t: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    labels: List[str]
    B: np.ndarray
    M: np.ndarray
    D: np.ndarray
    gen_bus_ids: List[int]


class AndesKundurRunner:
    """
    Loads ANDES Kundur case, runs PF+TDS, extracts GENROU rotor states,
    and builds B using shortest-path reactance between generator buses.
    """

    def __init__(self, T: float = 20.0):
        self.T = float(T)

    @staticmethod
    def _try_get_var(model, candidates: List[str]) -> Any:
        for nm in candidates:
            if hasattr(model, nm):
                return getattr(model, nm)
        raise RuntimeError(
            f"Could not find any of variables {candidates} in model {type(model)}"
        )

    @staticmethod
    def _as_array(v) -> np.ndarray:
        if hasattr(v, "v"):
            return np.array(v.v)
        return np.array(v)

    @staticmethod
    def _extract_time_series(ss, var) -> np.ndarray:
        if not hasattr(var, "a"):
            raise RuntimeError("Expected ANDES variable with address attribute .a")
        return np.array(ss.dae.ts.x[:, var.a], dtype=float)

    @staticmethod
    def _extract_machine_labels(model, fallback_prefix: str = "G") -> List[str]:
        for key in ["idx", "name", "uid", "ID"]:
            if hasattr(model, key):
                vv = getattr(model, key)
                try:
                    arr = vv.v if hasattr(vv, "v") else vv
                    out = [str(x) for x in list(arr)]
                    if out:
                        return out
                except Exception:
                    pass
        try:
            n = int(model.n)
        except Exception:
            n = 4
        return [f"{fallback_prefix}{i+1}" for i in range(n)]

    @staticmethod
    def _build_bus_reactance_graph(ss) -> nx.Graph:
        """
        Build an undirected bus graph with edge weight 'x' = |reactance|.

        We try to be robust across ANDES versions by scanning possible model names
        and field names. For parallel elements between the same buses, we keep the
        minimum |x| (strongest corridor) as a conservative proxy for shortest-path.
        """
        # candidate model names that often contain branches/transformers
        model_names = [
            "Line",
            "line",
            "ACLine",
            "acline",
            "TLine",
            "tline",
            "Branch",
            "branch",
            "Xfmr",
            "xfmr",
            "Transformer",
            "transformer",
            "XFormer",
            "xformer",
        ]

        candidates = []
        for nm in model_names:
            if hasattr(ss, nm):
                candidates.append(getattr(ss, nm))

        if not candidates:
            raise RuntimeError(
                "Could not find any line/branch/xfmr model in ANDES system."
            )

        def _get_field(obj, names: List[str]) -> Optional[np.ndarray]:
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    try:
                        return np.array(v.v) if hasattr(v, "v") else np.array(v)
                    except Exception:
                        continue
            return None

        # field name candidates
        from_names = ["bus1", "from_bus", "fbus", "from", "fb", "busf", "f"]
        to_names = ["bus2", "to_bus", "tbus", "to", "tb", "bust", "t"]
        x_names = ["x", "x12", "xsc", "X", "reactance", "x_pu", "xpu"]

        # build graph
        G = nx.Graph()
        edge_min_x: Dict[Tuple[int, int], float] = {}

        for obj in candidates:
            fb = _get_field(obj, from_names)
            tb = _get_field(obj, to_names)
            xx = _get_field(obj, x_names)
            if fb is None or tb is None or xx is None:
                continue

            m = min(len(fb), len(tb), len(xx))
            for k in range(m):
                i = int(fb[k])
                j = int(tb[k])
                xk = float(xx[k])
                ax = abs(xk)
                if not np.isfinite(ax) or ax < 1e-9:
                    continue
                a, b = (i, j) if i <= j else (j, i)
                key = (a, b)
                if key not in edge_min_x or ax < edge_min_x[key]:
                    edge_min_x[key] = ax

        if not edge_min_x:
            raise RuntimeError(
                "No usable (from_bus,to_bus,x) edges found to build bus reactance graph. "
                "You may need to adapt field names for your ANDES version."
            )

        for (i, j), ax in edge_min_x.items():
            if i == j:
                continue
            G.add_edge(i, j, x=ax)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            raise RuntimeError("Bus reactance graph ended up empty.")

        return G

    @staticmethod
    def _build_B_from_shortest_path_reactance(ss, gen_bus_ids: List[int]) -> np.ndarray:
        """
        Robust dense proxy:
          B_ij = 1 / X_shortest_path(bus_i, bus_j), with edge weights = |x|.
        """
        G = AndesKundurRunner._build_bus_reactance_graph(ss)

        gbus = [int(x) for x in gen_bus_ids]
        Ng = len(gbus)
        B = np.zeros((Ng, Ng), dtype=float)

        for a in range(Ng):
            for c in range(a + 1, Ng):
                s = gbus[a]
                t = gbus[c]
                try:
                    xsp = nx.shortest_path_length(G, source=s, target=t, weight="x")
                except nx.NetworkXNoPath:
                    xsp = float("inf")
                if not np.isfinite(xsp) or xsp <= 0.0:
                    continue
                B[a, c] = B[c, a] = float(1.0 / max(1e-12, xsp))

        return B

    def run(self) -> AndesSimOut:
        try:
            import andes  # type: ignore
            from andes.utils.paths import get_case  # type: ignore
        except Exception as e:
            raise RuntimeError("ANDES not installed. Run: pip install andes") from e

        case_rel = os.environ.get("ANDES_CASE", "kundur/kundur_full.xlsx")
        ss = andes.run(get_case(case_rel))

        # set horizon if available
        if (
            hasattr(ss, "TDS")
            and hasattr(ss.TDS, "config")
            and hasattr(ss.TDS.config, "tf")
        ):
            ss.TDS.config.tf = float(self.T)

        # run TDS
        if hasattr(ss, "TDS") and hasattr(ss.TDS, "run"):
            ss.TDS.run()

        if (
            not hasattr(ss, "dae")
            or not hasattr(ss.dae, "ts")
            or not hasattr(ss.dae.ts, "t")
        ):
            raise RuntimeError(
                "ANDES time series not found. Expect ss.dae.ts.t after TDS.run()."
            )
        t = np.array(ss.dae.ts.t, dtype=float)

        if not hasattr(ss, "GENROU"):
            raise RuntimeError("Expected ss.GENROU in Kundur case (GENROU machines).")
        gen = ss.GENROU

        labels = self._extract_machine_labels(gen, fallback_prefix="G")

        omega_var = self._try_get_var(gen, ["omega", "w"])
        delta_var = self._try_get_var(gen, ["delta", "delta0", "ang", "angle"])
        omega = self._extract_time_series(ss, omega_var)
        delta = self._extract_time_series(ss, delta_var)

        if not hasattr(gen, "bus"):
            raise RuntimeError("Expected generator bus mapping at ss.GENROU.bus.")
        gen_bus_ids = list(self._as_array(gen.bus).astype(int))

        # inertia/damping proxies
        if hasattr(gen, "H"):
            H = self._as_array(gen.H).astype(float)
            M = 2.0 * np.maximum(1e-6, H)  # toy scaling
        elif hasattr(gen, "M"):
            M = self._as_array(gen.M).astype(float)
        else:
            M = np.ones(omega.shape[1], dtype=float) * 6.0

        if hasattr(gen, "D"):
            D = self._as_array(gen.D).astype(float)
        else:
            D = np.ones(omega.shape[1], dtype=float) * 0.0

        # robust dense coupling proxy
        B = self._build_B_from_shortest_path_reactance(ss, gen_bus_ids=gen_bus_ids)

        # sanity checks
        Ng = omega.shape[1]
        if delta.shape[1] != Ng or B.shape != (Ng, Ng) or M.size != Ng or D.size != Ng:
            raise RuntimeError(
                f"Dimension mismatch: omega={omega.shape}, delta={delta.shape}, B={B.shape}, M={M.shape}, D={D.shape}"
            )

        # gauge angles to zero-mean per time sample
        delta = delta - delta.mean(axis=1, keepdims=True)

        return AndesSimOut(
            t=t,
            delta=delta,
            omega=omega,
            labels=labels,
            B=B,
            M=M,
            D=D,
            gen_bus_ids=gen_bus_ids,
        )


# ============================================================
# Plotting
# ============================================================
class Plotter:
    @staticmethod
    def _save(fig: plt.Figure, outbase: str) -> None:
        fig.tight_layout()
        fig.savefig(outbase + ".png", bbox_inches="tight")
        fig.savefig(outbase + ".pdf", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_time_series(
        figs_dir: str,
        labels: List[str],
        t: np.ndarray,
        omega: np.ndarray,
        delta: np.ndarray,
        ecfg: EventConfig,
    ) -> None:
        # omega(t)
        fig = plt.figure(figsize=(6.6, 2.6))
        for i, name in enumerate(labels):
            plt.plot(t, omega[:, i], label=name)
        plt.xlabel("t [s]")
        plt.ylabel(r"$\omega$ (pu)")
        plt.title("Kundur (ANDES): rotor speed states")
        plt.legend(ncol=4, frameon=False)
        Plotter._save(fig, os.path.join(figs_dir, "ts_omega"))

        # delta(t)
        fig = plt.figure(figsize=(6.6, 2.6))
        for i, name in enumerate(labels):
            plt.plot(t, delta[:, i], label=name)
        plt.xlabel("t [s]")
        plt.ylabel(r"$\delta$ [rad] (shifted to mean=0)")
        plt.title("Kundur (ANDES): rotor angle states")
        plt.legend(ncol=4, frameon=False)
        Plotter._save(fig, os.path.join(figs_dir, "ts_delta"))

        # max separation
        max_sep = np.max(delta, axis=1) - np.min(delta, axis=1)
        fig = plt.figure(figsize=(6.6, 2.4))
        plt.plot(t, max_sep, label=r"$\max(\delta)-\min(\delta)$")
        plt.axhline(ecfg.delta_sep_trip_rad, linestyle=":", linewidth=1.0, label="trip")
        plt.xlabel("t [s]")
        plt.ylabel("angle separation [rad]")
        plt.title("Loss-of-synchronism proxy")
        plt.legend(frameon=False)
        Plotter._save(fig, os.path.join(figs_dir, "ts_angle_separation"))

    @staticmethod
    def plot_geometry_panel(
        figs_dir: str, labels: List[str], out: Dict[str, Any]
    ) -> None:
        geom = out["diagnostics"]["operating_point_geometry_by_window"]["precrisis"]
        energy = np.array(geom["critical_mode_eff"]["node_energy"], dtype=float)

        frag = geom["forman_ricci_eff"]["node_fragility_min_incident_curv"]
        frag_v = np.array([float(frag.get(k, np.nan)) for k in labels], dtype=float)

        bott = geom["forman_ricci_eff"]["bottleneck_edges"]
        bott_txt = ", ".join(
            [f"{b['edge'][0]}-{b['edge'][1]}:{b['curv']:.2f}" for b in bott]
        )

        fig = plt.figure(figsize=(6.8, 3.2))

        ax1 = plt.subplot(1, 2, 1)
        ax1.bar(labels, energy)
        ax1.set_title("Critical-mode participation")
        ax1.set_ylabel("node energy")
        ax1.set_xlabel("node")

        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(labels, -frag_v)  # plot (-curv) as fragility magnitude
        ax2.set_title("Fragility (−min incident curvature)")
        ax2.set_ylabel("−curvature")
        ax2.set_xlabel("node")

        fig.suptitle(f"Precrisis geometry. Bottlenecks: {bott_txt}", y=1.02)
        Plotter._save(fig, os.path.join(figs_dir, "geom_precrisis_panel"))

    @staticmethod
    def plot_window_summary(figs_dir: str, out: Dict[str, Any]) -> None:
        sota = out["diagnostics"]["sota_indicators_by_window"]
        geom = out["diagnostics"]["operating_point_geometry_by_window"]

        windows = ["pre", "post", "precrisis"]
        lam2 = [geom[w]["lambda2_H_eff"] for w in windows]
        mincos = [geom[w]["min_cos_on_edges"] for w in windows]
        rop95 = [sota[w]["rocof"]["abs_p95"] for w in windows]
        sep = [sota[w]["angle_separation"]["max_delta_sep_rad"] for w in windows]

        x = np.arange(len(windows))

        fig = plt.figure(figsize=(6.8, 3.2))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(x, lam2, marker="o", label=r"$\lambda_2(H_\mathrm{eff})$")
        ax1.plot(x, mincos, marker="s", label=r"$\min \cos(\Delta\delta^*)$")
        ax1.set_xticks(x, windows)
        ax1.set_title("Geometry indicators (operating-point dependent)")
        ax1.legend(frameon=False)

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(x, rop95, marker="o", label="RoCoF |p95|")
        ax2.plot(x, sep, marker="s", label="max angle sep")
        ax2.set_xticks(x, windows)
        ax2.set_title("SOTA-like indicators (window summaries)")
        ax2.legend(frameon=False)

        Plotter._save(fig, os.path.join(figs_dir, "window_summary"))


# ============================================================
# Experiment runner (ANDES-backed)
# ============================================================
class HiddenMarginQ1ExperimentANDES:
    def __init__(self, paths: Paths, gcfg: GraphConfig, ecfg: EventConfig):
        self.paths = paths
        self.gcfg = gcfg
        self.ecfg = ecfg
        IO.ensure_dir(paths.results_dir)
        IO.ensure_dir(paths.figs_dir)

    @staticmethod
    def _delta_star_from_window(
        t: np.ndarray, delta: np.ndarray, t0: float, t1: float
    ) -> np.ndarray:
        m = Windows.mask(t, t0, t1)
        seg = delta[m, :]
        if seg.size == 0:
            return np.zeros(delta.shape[1], dtype=float)
        return np.mean(seg, axis=0)

    @staticmethod
    def _estimate_dt(t: np.ndarray, fallback: float) -> float:
        if t.size < 3:
            return float(fallback)
        d = np.diff(t)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return float(fallback)
        return float(np.median(d))

    @staticmethod
    def _omega_ref_from_prewindow(
        t: np.ndarray, omega: np.ndarray, win_pre: Tuple[float, float]
    ) -> np.ndarray:
        m = Windows.mask(t, win_pre[0], win_pre[1])
        seg = omega[m, :]
        if seg.size == 0:
            return omega[0, :]
        return np.median(seg, axis=0)

    def run(self) -> Tuple[Dict[str, Any], AndesSimOut]:
        # 1) Run ANDES Kundur
        sim = AndesKundurRunner(T=self.ecfg.T).run()

        t = sim.t
        delta = sim.delta
        omega = sim.omega
        labels = sim.labels
        B = sim.B
        M = sim.M
        D = sim.D

        dt = self._estimate_dt(t, self.ecfg.dt_guess)

        # 2) Baseline geometry (pure B): H0 = M^{-1/2} L(B) M^{-1/2}
        L0 = Lap.laplacian_from_weights(B)
        H0 = Spectral.inertial_laplacian(L0, M)
        lam2_H0 = Spectral.lambda2(H0)

        # 3) Windows
        w_pre = self.ecfg.win_pre
        w_post = self.ecfg.win_post
        w_precrisis = self.ecfg.win_precrisis
        w_crisis = self.ecfg.win_crisis

        # 4) SOTA window summaries (from delta/omega)
        sota = {
            "pre": SOTAIndicators.summarize(t, delta, omega, dt, w_pre[0], w_pre[1]),
            "post": SOTAIndicators.summarize(t, delta, omega, dt, w_post[0], w_post[1]),
            "precrisis": SOTAIndicators.summarize(
                t, delta, omega, dt, w_precrisis[0], w_precrisis[1]
            ),
            "crisis": SOTAIndicators.summarize(
                t, delta, omega, dt, w_crisis[0], w_crisis[1]
            ),
        }

        # 5) Operating points δ*
        delta_pre_star = self._delta_star_from_window(t, delta, w_pre[0], w_pre[1])
        delta_post_star = self._delta_star_from_window(t, delta, w_post[0], w_post[1])
        delta_precrisis_star = self._delta_star_from_window(
            t, delta, w_precrisis[0], w_precrisis[1]
        )

        geom_by_window = {
            "pre": OperatingPointGeometry.summarize(labels, B, M, delta_pre_star),
            "post": OperatingPointGeometry.summarize(labels, B, M, delta_post_star),
            "precrisis": OperatingPointGeometry.summarize(
                labels, B, M, delta_precrisis_star
            ),
        }

        # 6) Crisis detection (omega deviation-based)
        omega_ref = self._omega_ref_from_prewindow(t, omega, win_pre=w_pre)
        crisis = CrisisDetector.detect(
            t=t,
            delta=delta,
            omega=omega,
            omega_ref=omega_ref,
            dt=dt,
            delta_sep_trip_rad=self.ecfg.delta_sep_trip_rad,
            omega_dev_trip=self.ecfg.omega_dev_trip,
            rocof_trip=self.ecfg.rocof_trip,
            sustain_sec=self.ecfg.crisis_sustain_sec,
        )

        # 7) Warn logic (precrisis)
        pre_ro_p95 = float(sota["precrisis"]["rocof"]["abs_p95"])
        pre_sep = float(sota["precrisis"]["angle_separation"]["max_delta_sep_rad"])
        # omega dev warn: use deviation from reference, not absolute omega
        mpre = Windows.mask(t, w_precrisis[0], w_precrisis[1])
        if np.any(mpre):
            om_dev = omega[mpre, :] - omega_ref.reshape(1, -1)
            pre_om_dev_max = float(np.max(np.abs(om_dev)))
        else:
            pre_om_dev_max = float("nan")

        sota_warn = bool(
            (np.isfinite(pre_ro_p95) and pre_ro_p95 >= 0.8 * self.ecfg.rocof_trip)
            or (
                np.isfinite(pre_om_dev_max)
                and pre_om_dev_max >= 0.8 * self.ecfg.omega_dev_trip
            )
            or (np.isfinite(pre_sep) and pre_sep >= 0.8 * self.ecfg.delta_sep_trip_rad)
        )

        lam2_eff_pre = float(geom_by_window["pre"]["lambda2_H_eff"])
        lam2_eff_precrisis = float(geom_by_window["precrisis"]["lambda2_H_eff"])
        min_cos_precrisis = float(geom_by_window["precrisis"]["min_cos_on_edges"])
        geom_warn = bool(
            (lam2_eff_precrisis <= self.ecfg.lam2_eff_warn)
            or (min_cos_precrisis <= self.ecfg.min_cos_warn)
            or (lam2_eff_precrisis <= 0.5 * max(1e-12, lam2_eff_pre))
        )

        # 8) Hidden margin indices (normalized)
        step_amp_surrogate = 1.0
        hmi0 = float(step_amp_surrogate / max(1e-12, lam2_H0))
        hmi_eff = float(step_amp_surrogate / max(1e-12, lam2_eff_precrisis))

        # 9) Critical node rank: Fiedler energy + small curvature term
        energy = geom_by_window["precrisis"]["critical_mode_eff"]["node_energy"]
        frag = geom_by_window["precrisis"]["forman_ricci_eff"][
            "node_fragility_min_incident_curv"
        ]
        crit_score: Dict[str, float] = {}
        for i, name in enumerate(labels):
            f = float(frag.get(name, 0.0))
            crit_score[name] = float(energy[i]) + 0.05 * (-f)
        critical_rank = sorted(crit_score.items(), key=lambda kv: kv[1], reverse=True)

        out = {
            "meta": {
                "experiment": "kundur_hidden_margin_q1_andes",
                "notes": [
                    "Plant is ANDES Kundur case (kundur_full.xlsx).",
                    "Extracted GENROU rotor states from ss.dae.ts.",
                    "Built coupling B_ij = 1 / X_shortest_path(bus_i, bus_j) from network reactances (robust; dense proxy).",
                    "Operating-point geometry uses W_eff = B*cos(Δδ*).",
                    "Indices are scale-free here (step_amp surrogate=1.0).",
                ],
                "paths": {"results_dir": self.paths.results_dir},
            },
            "graph": {
                "nodes": labels,
                "gen_bus_ids": sim.gen_bus_ids,
                "B_matrix_proxy": B,
            },
            "system": {
                "M_proxy": M,
                "D_proxy": D,
                "baseline": {
                    "H0": H0,
                    "lambda2_H0": float(lam2_H0),
                },
                "dt_est": float(dt),
                "T": float(self.ecfg.T),
            },
            "windows": {
                "pre": list(map(float, w_pre)),
                "post": list(map(float, w_post)),
                "precrisis": list(map(float, w_precrisis)),
                "crisis": list(map(float, w_crisis)),
            },
            "diagnostics": {
                "sota_indicators_by_window": sota,
                "operating_point_geometry_by_window": geom_by_window,
                "hidden_margin_indices": {
                    "HMI0": {
                        "definition": "HMI0 = 1 / lambda2(H0) (normalized)",
                        "value": hmi0,
                        "lambda2_H0": float(lam2_H0),
                    },
                    "HMI_eff_precrisis": {
                        "definition": "HMI_eff = 1 / lambda2(H_eff(precrisis)) (normalized)",
                        "value": hmi_eff,
                        "lambda2_H_eff_precrisis": float(lam2_eff_precrisis),
                    },
                    "min_cos_precrisis": float(min_cos_precrisis),
                },
                "geometry_warn_thresholds": {
                    "lam2_eff_warn": float(self.ecfg.lam2_eff_warn),
                    "min_cos_warn": float(self.ecfg.min_cos_warn),
                },
                "crisis_detection": crisis,
                "did_sota_warn_precrisis": bool(sota_warn),
                "did_geometry_warn_precrisis": bool(geom_warn),
                "critical_nodes_rank_precrisis": [
                    {"node": k, "score": v} for k, v in critical_rank
                ],
            },
        }

        return out, sim

    def save(self, out: Dict[str, Any]) -> str:
        path = os.path.join(self.paths.results_dir, "diagnostic.json")
        IO.save_json(path, out)
        return path


# ============================================================
# main
# ============================================================
def main() -> None:
    set_ieee_style()

    paths = Paths.from_script(
        __file__, results_folder="results_kundur2_hidden_margin_q1_andes"
    )
    IO.ensure_dir(paths.results_dir)
    IO.ensure_dir(paths.figs_dir)

    gcfg = GraphConfig(w_intra=8.0, w_tie=0.6)
    ecfg = EventConfig(
        T=20.0,
        dt_guess=1.0 / 30.0,
        win_pre=(0.5, 3.0),
        win_post=(4.0, 8.0),
        win_precrisis=(8.5, 12.0),
        win_crisis=(12.0, 18.0),
        lam2_eff_warn=0.25,
        min_cos_warn=0.20,
        delta_sep_trip_rad=1.45,
        omega_dev_trip=0.01,  # <-- IMPORTANT: deviation threshold, not absolute omega
        rocof_trip=0.25,
        crisis_sustain_sec=0.5,
    )

    exp = HiddenMarginQ1ExperimentANDES(paths=paths, gcfg=gcfg, ecfg=ecfg)
    out, sim = exp.run()
    json_path = exp.save(out)

    print(f"[OK] Wrote diagnostic JSON: {json_path}")

    # Plots
    Plotter.plot_time_series(
        paths.figs_dir, sim.labels, sim.t, sim.omega, sim.delta, ecfg
    )
    Plotter.plot_geometry_panel(paths.figs_dir, sim.labels, out)
    Plotter.plot_window_summary(paths.figs_dir, out)

    diag = out["diagnostics"]
    print(f"did_sota_warn_precrisis     = {diag['did_sota_warn_precrisis']}")
    print(f"did_geometry_warn_precrisis = {diag['did_geometry_warn_precrisis']}")
    print(
        f"crisis_detected             = {diag['crisis_detection']['crisis_detected']}"
    )
    print(
        f"lambda2_H_eff(precrisis)    = {diag['operating_point_geometry_by_window']['precrisis']['lambda2_H_eff']:.6f}"
    )
    print(
        f"min_cos(precrisis)          = {diag['operating_point_geometry_by_window']['precrisis']['min_cos_on_edges']:.6f}"
    )
    print(
        f"HMI_eff(precrisis)          = {diag['hidden_margin_indices']['HMI_eff_precrisis']['value']:.6f}"
    )
    print("[INFO] Critical nodes (precrisis rank):")
    for r in diag["critical_nodes_rank_precrisis"][: min(6, len(sim.labels))]:
        print(f"  - {r['node']}: score={r['score']:.6f}")

    print(f"[OK] Saved figures to: {paths.figs_dir}")


if __name__ == "__main__":
    main()
