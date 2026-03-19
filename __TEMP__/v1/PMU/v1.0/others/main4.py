#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kundur Two-Area (4-machine) — Spectral–Lyapunov toolbox demo (refactored)
========================================================================

Clean OOP + DRY/KISS version of your script.

Pipeline:
1) Build simplified Kundur 2-area graph (4 generators) -> Laplacian L.
2) Simulate stochastic swing dynamics with OU disturbances (Euler–Maruyama).
3) Metrics:
   - Early warning: W(t)=tr(Cov(ω)) in sliding windows
   - Localization: participation π vs Var(ω_i)
4) Placement:
   - Greedy inertia placement minimizing tr(P_ωω) from continuous-time Lyapunov.

Outputs:
- results_kundur2/plots/*.png
- results_kundur2/results.json
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve_continuous_lyapunov


# ============================================================
# I/O helpers
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
    plots_dir: str

    @staticmethod
    def from_script(__file__: str, results_folder: str = "results_kundur2") -> "Paths":
        base = os.path.dirname(os.path.abspath(__file__))
        results = os.path.join(base, results_folder)
        plots = os.path.join(results, "plots")
        return Paths(base_dir=base, results_dir=results, plots_dir=plots)


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.01
    T: float = 120.0
    ou_gamma: float = 1.0
    noise_sigma: float = 0.10
    seed: int = 42

    # for plots / metrics
    win_sec: float = 10.0
    step_sec: float = 1.0

    # for json downsample
    ds: int = 10

    # if True: project to zero-mean subspace each step (removes mode-0 drift)
    project_zero_mean: bool = True


@dataclass(frozen=True)
class PlacementConfig:
    budget: float = 4.0
    step: float = 0.25
    caps: Tuple[float, float, float, float] = (4.0, 4.0, 4.0, 4.0)


@dataclass(frozen=True)
class KundurWeights:
    w_intra: float = 8.0
    w_tie: float = 2.0


# ============================================================
# Graph / Laplacian
# ============================================================
@dataclass(frozen=True)
class GraphSpec:
    nodes: List[str]
    edges: List[Tuple[str, str, float]]
    pos: Dict[str, Tuple[float, float]]


class Kundur2AreaGraph:
    """Builds a simplified 4-generator Kundur two-area graph and Laplacian."""

    def __init__(self, weights: KundurWeights = KundurWeights()):
        self.weights = weights

    def build_spec(self) -> GraphSpec:
        nodes = ["G1", "G2", "G3", "G4"]
        w_intra = float(self.weights.w_intra)
        w_tie = float(self.weights.w_tie)

        edges = [
            ("G1", "G2", w_intra),
            ("G3", "G4", w_intra),
            ("G2", "G3", w_tie),
            ("G1", "G4", w_tie),
        ]

        pos = {
            "G1": (-1.0, 0.4),
            "G2": (-1.0, -0.4),
            "G3": (1.0, 0.4),
            "G4": (1.0, -0.4),
        }
        return GraphSpec(nodes=nodes, edges=edges, pos=pos)

    @staticmethod
    def to_graph_and_laplacian(
        spec: GraphSpec,
    ) -> Tuple[nx.Graph, np.ndarray, List[str]]:
        G = nx.Graph()
        for n in spec.nodes:
            G.add_node(n)
        for u, v, w in spec.edges:
            G.add_edge(u, v, weight=float(w))

        labels = list(spec.nodes)
        n = len(labels)
        idx = {labels[i]: i for i in range(n)}

        L = np.zeros((n, n), dtype=float)
        for u, v, data in G.edges(data=True):
            w = float(data["weight"])
            i, j = idx[u], idx[v]
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w

        return G, L, labels


# ============================================================
# Math / Metrics
# ============================================================
class Spectral:
    @staticmethod
    def compute_H(L: np.ndarray, M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        inv_sqrt = np.diag(1.0 / np.sqrt(M))
        return inv_sqrt @ L @ inv_sqrt

    @staticmethod
    def spectral_gap(H: np.ndarray, tol: float = 1e-10) -> float:
        w = np.sort(np.linalg.eigvalsh(H))
        for val in w:
            if val > tol:
                return float(val)
        return 0.0

    @staticmethod
    def eig_smallest_nonzero(
        H: np.ndarray, k: int = 3, tol: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray]:
        w, V = np.linalg.eigh(H)
        order = np.argsort(w)
        w = w[order]
        V = V[:, order]
        keep = [i for i in range(len(w)) if w[i] > tol][:k]
        return w[keep], V[:, keep]


class Stats:
    @staticmethod
    def tr_cov(x: np.ndarray) -> float:
        # x: (T,n)
        C = np.cov(x.T, bias=True)
        return float(np.trace(C))

    @staticmethod
    def moving_cov_trace(
        x: np.ndarray, win: int, step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = x.shape[0]
        idxs, vals = [], []
        for start in range(0, T - win + 1, step):
            seg = x[start : start + win, :]
            idxs.append(start + win - 1)
            vals.append(Stats.tr_cov(seg))
        return np.asarray(idxs, dtype=int), np.asarray(vals, dtype=float)

    @staticmethod
    def tail_mean(x: np.ndarray, frac_tail: float = 0.2) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return float("nan")
        k0 = int((1.0 - frac_tail) * len(x))
        k0 = max(0, min(k0, len(x) - 1))
        return float(np.mean(x[k0:]))


class Participation:
    @staticmethod
    def scores(H: np.ndarray, omega_ss: np.ndarray, k_modes: int = 2) -> Dict[str, Any]:
        evals, evecs = Spectral.eig_smallest_nonzero(H, k=max(k_modes, 1))
        if len(evals) == 0:
            raise ValueError(
                "H has no nonzero eigenvalues (disconnected graph or numerical issue)."
            )

        P = np.cov(omega_ss.T, bias=True)

        # modal weights: energy along eigenvectors
        wts = np.zeros(evecs.shape[1], dtype=float)
        for j in range(evecs.shape[1]):
            u = evecs[:, j : j + 1]
            wts[j] = max(float((u.T @ P @ u).item()), 0.0)

        # participation
        pi = np.zeros(H.shape[0], dtype=float)
        for j in range(evecs.shape[1]):
            pi += wts[j] * (evecs[:, j] ** 2)

        return {"eigvals": evals, "weights": wts, "pi": pi}


# ============================================================
# Swing simulator (OU + projection)
# ============================================================
@dataclass(frozen=True)
class SimResult:
    t: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    w: np.ndarray


class SwingSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def run(self, L: np.ndarray, M: np.ndarray, D: np.ndarray) -> SimResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        L = np.asarray(L, dtype=float)
        M = np.asarray(M, dtype=float)
        D = np.asarray(D, dtype=float)

        n = L.shape[0]
        steps = int(cfg.T / cfg.dt) + 1
        t = np.linspace(0.0, cfg.T, steps)

        delta = np.zeros((steps, n), dtype=float)
        omega = np.zeros((steps, n), dtype=float)
        w = np.zeros((steps, n), dtype=float)

        Minv = 1.0 / M
        gamma = float(cfg.ou_gamma)
        sigma = float(cfg.noise_sigma)
        sqrt_dt = math.sqrt(cfg.dt)
        ou_k = math.sqrt(2.0 * gamma) * sigma

        for k in range(steps - 1):
            # OU update
            dW = rng.standard_normal(n) * sqrt_dt
            w[k + 1] = w[k] + (-gamma * w[k]) * cfg.dt + ou_k * dW

            if cfg.project_zero_mean:
                w[k + 1] -= w[k + 1].mean()

            # swing update
            delta[k + 1] = delta[k] + omega[k] * cfg.dt
            accel = (-D * omega[k] - L @ delta[k] + w[k]) * Minv
            omega[k + 1] = omega[k] + accel * cfg.dt

            if cfg.project_zero_mean:
                delta[k + 1] -= delta[k + 1].mean()
                omega[k + 1] -= omega[k + 1].mean()

        return SimResult(t=t, delta=delta, omega=omega, w=w)


# ============================================================
# Lyapunov placement objective (continuous-time)
# ============================================================
class LyapunovObjective:
    @staticmethod
    def build_A(L: np.ndarray, M: np.ndarray, D: np.ndarray) -> np.ndarray:
        n = L.shape[0]
        Minv = np.diag(1.0 / M)
        Z = np.zeros((n, n))
        I = np.eye(n)
        return np.block([[Z, I], [-Minv @ L, -Minv @ np.diag(D)]])

    @staticmethod
    def trace_Pww(L: np.ndarray, M: np.ndarray, D: np.ndarray, sigma_w: float) -> float:
        n = L.shape[0]
        A = LyapunovObjective.build_A(L, M, D)

        # noise enters omega_dot through Minv
        B = np.block([[np.zeros((n, n))], [np.diag(1.0 / M)]])

        Q = (sigma_w**2) * (B @ B.T)
        P = solve_continuous_lyapunov(A, -Q)  # A P + P A^T + Q = 0
        Pww = P[n:, n:]
        return float(np.trace(Pww))


class GreedyPlacement:
    """Greedy inertia placement under caps and budget minimizing tr(Pww)."""

    def __init__(self, sigma_w: float):
        self.sigma_w = float(sigma_w)

    def solve(
        self,
        L: np.ndarray,
        D: np.ndarray,
        M0: np.ndarray,
        budget: float,
        step: float,
        caps: np.ndarray,
    ) -> Dict[str, Any]:
        M0 = np.asarray(M0, dtype=float)
        D = np.asarray(D, dtype=float)
        caps = np.asarray(caps, dtype=float)

        n = len(M0)
        dM = np.zeros(n, dtype=float)
        remaining = float(budget)

        def J(M: np.ndarray) -> float:
            return LyapunovObjective.trace_Pww(L, M, D, self.sigma_w)

        J_base = J(M0)

        while remaining > 1e-12:
            inc = min(step, remaining)
            current = J(M0 + dM)

            best_i: Optional[int] = None
            best_val = float("inf")

            for i in range(n):
                if dM[i] + inc > caps[i] + 1e-12:
                    continue
                M_try = (M0 + dM).copy()
                M_try[i] += inc
                val = J(M_try)
                if val < best_val:
                    best_val = val
                    best_i = i

            if best_i is None:
                break

            # stop if no meaningful improvement
            if best_val >= current - 1e-12:
                break

            dM[best_i] += inc
            remaining -= inc

        J_final = J(M0 + dM)
        return {
            "dM": dM,
            "budget_used": float(budget - remaining),
            "J_base_trPww": float(J_base),
            "J_final_trPww": float(J_final),
            "J_improvement": float(J_base - J_final),
        }


# ============================================================
# Plotting
# ============================================================
class Plotter:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir

    def _save(self, fname: str) -> str:
        return os.path.join(self.plots_dir, fname)

    def save_graph(
        self,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        fname: str = "kundur2area_graph.png",
    ) -> None:
        plt.figure(figsize=(6.2, 3.0))
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=12)
        nx.draw_networkx_edges(G, pos, width=[1.5 + 0.6 * w for w in weights])
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self._save(fname), dpi=200)
        plt.close()

    def early_warning(
        self, t: np.ndarray, omega: np.ndarray, cfg: SimConfig, title: str, prefix: str
    ) -> Dict[str, Any]:
        # NOTE: mean omega often ~0 (especially with projection). Keep for completeness.
        mean_omega = omega.mean(axis=1)

        win = max(5, int(cfg.win_sec / cfg.dt))
        step = max(1, int(cfg.step_sec / cfg.dt))
        idxs, trP = Stats.moving_cov_trace(omega, win=win, step=step)
        tt = t[idxs]

        plt.figure(figsize=(8.5, 4.2))
        plt.plot(t, mean_omega, linewidth=1.6)
        plt.xlabel("time (s)")
        plt.ylabel("mean frequency deviation (pu)")
        plt.title(title + " — mean ω")
        plt.tight_layout()
        plt.savefig(self._save(f"{prefix}_mean_omega.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8.5, 4.2))
        plt.plot(tt, trP, linewidth=1.8)
        plt.xlabel("time (s)")
        plt.ylabel("tr(Cov(ω)) (windowed)")
        plt.title(title + " — early warning index W(t)=tr(Cov(ω))")
        plt.tight_layout()
        plt.savefig(self._save(f"{prefix}_warning_trace.png"), dpi=200)
        plt.close()

        return {"t_win": tt, "trace_cov_omega": trP, "mean_omega": mean_omega}

    def localization(
        self,
        labels: List[str],
        pi: np.ndarray,
        var_omega: np.ndarray,
        title: str,
        prefix: str,
    ) -> None:
        x = np.arange(len(labels))

        plt.figure(figsize=(8.5, 4.2))
        plt.bar(x, pi)
        plt.xticks(x, labels)
        plt.ylabel("participation π_i")
        plt.title(title + " — localization by participation")
        plt.tight_layout()
        plt.savefig(self._save(f"{prefix}_pi.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8.5, 4.2))
        plt.bar(x, var_omega)
        plt.xticks(x, labels)
        plt.ylabel("Var(ω_i)")
        plt.title(title + " — measured frequency variance")
        plt.tight_layout()
        plt.savefig(self._save(f"{prefix}_varomega.png"), dpi=200)
        plt.close()

    def placement_summary(
        self,
        lam2_base: float,
        lam2_final: float,
        W_base: float,
        W_after: float,
        title: str,
    ) -> None:
        plt.figure(figsize=(8.5, 4.2))
        plt.bar([0, 1], [lam2_base, lam2_final])
        plt.xticks([0, 1], ["base", "after placement"])
        plt.ylabel("λ2(H)")
        plt.title(title + " — spectral gap (note: not monotone in M)")
        plt.tight_layout()
        plt.savefig(self._save("placement_summary_lambda2.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8.5, 4.2))
        plt.bar([0, 1], [W_base, W_after])
        plt.xticks([0, 1], ["base", "after placement"])
        plt.ylabel("tail-mean W(t)=tr(Cov(ω))")
        plt.title(title + " — variance proxy reduction")
        plt.tight_layout()
        plt.savefig(self._save("placement_summary_trcov_reduction.png"), dpi=200)
        plt.close()


# ============================================================
# Experiment runner
# ============================================================
@dataclass
class ScenarioResult:
    name: str
    M: np.ndarray
    D: np.ndarray
    lambda2_H: float
    early_warning: Dict[str, Any]
    localization: Dict[str, Any]
    raw: Dict[str, Any]


class KundurExperiment:
    def __init__(self, paths: Paths, sim_cfg: SimConfig):
        self.paths = paths
        self.sim_cfg = sim_cfg

        IO.ensure_dir(paths.results_dir)
        IO.ensure_dir(paths.plots_dir)

        self.plotter = Plotter(paths.plots_dir)
        self.sim = SwingSimulator(sim_cfg)

    def run_scenario(
        self,
        name: str,
        L: np.ndarray,
        labels: List[str],
        M: np.ndarray,
        D: np.ndarray,
        prefix: str,
    ) -> ScenarioResult:
        H = Spectral.compute_H(L, M)
        lam2 = Spectral.spectral_gap(H)

        sim = self.sim.run(L=L, M=M, D=D)
        omega = sim.omega

        ew = self.plotter.early_warning(
            t=sim.t,
            omega=omega,
            cfg=self.sim_cfg,
            title=f"{name} (λ2(H)={lam2:.4f})",
            prefix=prefix,
        )

        half = omega.shape[0] // 2
        omega_ss = omega[half:, :]
        var_omega = np.var(omega_ss, axis=0)

        part = Participation.scores(H, omega_ss, k_modes=2)
        self.plotter.localization(
            labels=labels,
            pi=part["pi"],
            var_omega=var_omega,
            title=f"{name} (localization)",
            prefix=prefix,
        )

        ds = max(1, int(self.sim_cfg.ds))
        raw = {"t_ds": sim.t[::ds], "omega_ds": omega[::ds, :]}

        return ScenarioResult(
            name=name,
            M=M.copy(),
            D=D.copy(),
            lambda2_H=lam2,
            early_warning=ew,
            localization={
                "pi": part["pi"],
                "var_omega": var_omega,
                "eigvals_small": part["eigvals"],
                "mode_weights": part["weights"],
            },
            raw=raw,
        )


# ============================================================
# main
# ============================================================
def main():
    paths = Paths.from_script(__file__, results_folder="results_kundur2")
    sim_cfg = SimConfig(
        dt=0.01,
        T=120.0,
        ou_gamma=1.0,
        noise_sigma=0.10,
        seed=42,
        project_zero_mean=True,
    )
    placement_cfg = PlacementConfig(budget=4.0, step=0.25, caps=(4.0, 4.0, 4.0, 4.0))

    # Build graph + Laplacian
    gb = Kundur2AreaGraph(weights=KundurWeights(w_intra=8.0, w_tie=2.0))
    spec = gb.build_spec()
    G, L, labels = Kundur2AreaGraph.to_graph_and_laplacian(spec)

    exp = KundurExperiment(paths=paths, sim_cfg=sim_cfg)
    exp.plotter.save_graph(G, spec.pos, fname="kundur2area_graph.png")

    # Base parameters
    M_base = np.array([6.0, 5.0, 6.0, 5.0], dtype=float)
    D_base = np.array([1.2, 1.0, 1.2, 1.0], dtype=float)

    # Scenario A
    A = exp.run_scenario(
        name="Scenario A: Base",
        L=L,
        labels=labels,
        M=M_base,
        D=D_base,
        prefix="A_base",
    )

    # Scenario B
    factor = 0.35
    M_low = M_base.copy()
    M_low[2] *= factor
    M_low[3] *= factor

    B = exp.run_scenario(
        name=f"Scenario B: Low inertia in Area B (x{factor:.2f})",
        L=L,
        labels=labels,
        M=M_low,
        D=D_base,
        prefix="B_low_inertia_areaB",
    )

    # Scenario C (placement) - minimize tr(Pww) via Lyapunov
    caps = np.array(placement_cfg.caps, dtype=float)
    placer = GreedyPlacement(sigma_w=sim_cfg.noise_sigma)
    placement = placer.solve(
        L=L,
        D=D_base,
        M0=M_low,
        budget=placement_cfg.budget,
        step=placement_cfg.step,
        caps=caps,
    )
    M_place = M_low + placement["dM"]

    C = exp.run_scenario(
        name=f"Scenario C: Placement on top of low-inertia (budget={placement_cfg.budget})",
        L=L,
        labels=labels,
        M=M_place,
        D=D_base,
        prefix="C_placement",
    )

    # Placement summary using tail-mean of W(t)
    W_base = Stats.tail_mean(B.early_warning["trace_cov_omega"], frac_tail=0.2)
    W_after = Stats.tail_mean(C.early_warning["trace_cov_omega"], frac_tail=0.2)

    exp.plotter.placement_summary(
        lam2_base=B.lambda2_H,
        lam2_final=C.lambda2_H,
        W_base=W_base,
        W_after=W_after,
        title="Placement effect (starting from low inertia in Area B)",
    )

    # Pack results
    results = {
        "meta": {
            "script": "kundur2area_toolbox_refactored.py",
            "notes": [
                "Simplified Kundur 2-area (4 machines) using swing equations + Laplacian coupling.",
                "OU-shaped noise; optional per-step projection to zero-mean subspace to remove mode-0 drift.",
                "Placement uses continuous-time Lyapunov objective tr(P_ww) (minimization).",
            ],
            "paths": {"results_dir": paths.results_dir, "plots_dir": paths.plots_dir},
        },
        "graph": {
            "nodes": labels,
            "edges": [
                {"u": u, "v": v, "w": float(G[u][v]["weight"])} for u, v in G.edges()
            ],
            "laplacian_L": L,
        },
        "configs": {
            "sim": asdict(sim_cfg),
            "placement": asdict(placement_cfg),
            "M_base": M_base,
            "D_base": D_base,
            "low_inertia_factor_areaB": factor,
        },
        "placement": placement,
        "scenarios": {
            "A_base": asdict(A),
            "B_low_inertia_areaB": asdict(B),
            "C_placement": asdict(C),
        },
    }

    IO.save_json(os.path.join(paths.results_dir, "results.json"), results)
    print(f"[OK] Saved results to: {paths.results_dir}")
    print(f"[OK] Plots in: {paths.plots_dir}")
    print(f"[OK] JSON: {os.path.join(paths.results_dir, 'results.json')}")


if __name__ == "__main__":
    main()
