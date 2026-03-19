#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e7_rls_ibr.py

E7 — RLS IBR-aware (harmonics + optional interharmonics + DC), strictly online.

Core idea
---------
We fit a *linear* amplitude model on a sinusoidal dictionary (fundamental harmonics + optional fixed
interharmonics + DC) using RLS, while the fundamental frequency f0 is *nonlinear*.

At each sample k:
1) Build a small candidate grid around previous f_hat.
2) Score each candidate using a *normalized innovation* cost:
       J(f) = e(f)^2 / S(f)  +  freq_penalty * (f - f_prev)^2  +  nom_penalty * (f - f_nom)^2
   where:
       e(f) = y[k] - phi(k,f)^T theta
       S(f) = lam + phi(k,f)^T P phi(k,f)          (innovation variance proxy)
   This is much more stable than plain e^2 (prevents weird collapses under interharmonics).
3) Commit best f, then do exactly ONE RLS update with that chosen phi.

Stability upgrades vs earlier version
-------------------------------------
- Use normalized innovation cost (e^2 / S) for frequency selection.
- Warmup period: lock to nominal for first N samples (avoids early-grid chaos).
- Joseph-form covariance update + optional symmetry enforcement for P.
- Avoid lam=1.0 in tuning grid (still allowed if user sets it manually).
- Optional prior to nominal (nom_penalty) to prevent drift-to-edge under interharmonics.

PFEBench integration
--------------------
- Inherits BaseEstimator
- Implements tuning_ranges(), reset(), latency_samples, _step()
"""

from __future__ import annotations

from typing import List
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class RecursiveLeastSquaresIBREstimator(BaseEstimator):
    NAME: str = "RecursiveLeastSquaresIBREstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        FAST grid (~432 combos). Enough to tune the real levers without exploding runtime.
        """
        return [
            TuningParam(
                name="lam",
                default=0.995,
                type="float",
                values=[0.99, 0.995, 0.998],
                description="RLS forgetting factor.",
            ),
            TuningParam(
                name="delta",
                default=1e3,
                type="float",
                values=[1e2, 1e3, 1e4],
                description="Initial covariance scale.",
            ),
            TuningParam(
                name="search_span_hz",
                default=1.0,
                type="float",
                values=[0.8, 1.0, 1.2],
                description="+/- span for f grid (Hz).",
            ),
            TuningParam(
                name="search_step_hz",
                default=0.05,
                type="float",
                values=[0.05, 0.10],
                description="Grid step (Hz).",
            ),
            TuningParam(
                name="freq_penalty",
                default=0.05,
                type="float",
                values=[0.0, 0.02, 0.05],
                description="Penalty on per-sample df.",
            ),
            TuningParam(
                name="nom_penalty",
                default=0.002,
                type="float",
                values=[0.0, 0.001, 0.002],
                description="Weak prior to f_nom.",
            ),
            # Keep these FIXED by default (not tuned):
            # warmup_samples=25, p_sym_every=25, min/max via params or scenario
        ]

    # -----------------------
    # Helpers
    # -----------------------
    def _param_dim(self) -> int:
        dim = 2 * len(self._harmonics) + 2 * len(self._interharmonics)
        if self._include_dc:
            dim += 1
        return dim

    def _phi(self, k: int, f0_hz: float) -> np.ndarray:
        Ts = 1.0 / self._fs
        t = k * Ts

        w0 = 2.0 * np.pi * f0_hz
        parts = []

        for m in self._harmonics:
            ang = (m * w0) * t
            parts.append(np.cos(ang))
            parts.append(np.sin(ang))

        for fi in self._interharmonics:
            ang = (2.0 * np.pi * fi) * t
            parts.append(np.cos(ang))
            parts.append(np.sin(ang))

        if self._include_dc:
            parts.append(1.0)

        return np.asarray(parts, dtype=float)

    def _candidate_grid(self, f_prev: float) -> np.ndarray:
        span = float(self._search_span)
        step = float(self._search_step)

        if span <= 0.0 or step <= 0.0:
            return np.array([f_prev], dtype=float)

        n_side = int(np.floor(span / step))
        grid = f_prev + step * np.arange(-n_side, n_side + 1, dtype=float)

        # hard bounds
        grid = np.clip(grid, self._min_f, self._max_f)

        # unique and sorted
        grid = np.unique(grid)
        return grid

    def _rls_update(self, phi: np.ndarray, y: float) -> float:
        """
        RLS amplitude update with a numerically safer (Joseph) covariance form.
        Returns the innovation e.
        """
        P = self._P
        lam = float(self._lam)

        P_phi = P @ phi
        S = lam + float(phi.T @ P_phi)
        if (not np.isfinite(S)) or S <= self.EPS:
            S = self.EPS

        K = P_phi / S

        y_hat = float(phi.T @ self._theta)
        e = float(y - y_hat)

        # theta
        self._theta = self._theta + K * e

        # Joseph form
        I = np.eye(P.shape[0], dtype=float)
        KH = np.outer(K, phi)
        P_new = (I - KH) @ P @ (I - KH).T + np.outer(K, K)  # assume meas noise var=1
        P_new = P_new / lam
        self._P = P_new

        # occasional symmetrization
        self._upd_count += 1
        if self._p_sym_every > 0 and (self._upd_count % self._p_sym_every == 0):
            self._P = 0.5 * (self._P + self._P.T)

        return e

    # -----------------------
    # Lifecycle
    # -----------------------
    def reset(self) -> None:
        self._fs = float(self._params.get("fs", self.DEFAULT_FS))
        self._f_nom = float(self._params.get("f_nom_hz", self.NOMINAL_FREQ_HZ))

        self._lam = float(self._params.get("lam", 0.995))
        self._delta = float(self._params.get("delta", 1e3))

        self._min_f = float(self._params.get("min_freq_hz", 30.0))
        self._max_f = float(self._params.get("max_freq_hz", 70.0))

        self._search_span = float(self._params.get("search_span_hz", 1.0))
        self._search_step = float(self._params.get("search_step_hz", 0.05))
        self._freq_penalty = float(self._params.get("freq_penalty", 0.05))
        self._nom_penalty = float(self._params.get("nom_penalty", 0.002))
        self._warmup = int(self._params.get("warmup_samples", 25))
        self._p_sym_every = int(self._params.get("p_sym_every", 25))

        # Structure params (can be overridden via method config)
        self._harmonics = tuple(self._params.get("harmonics", (1, 2, 3, 5, 7)))
        self._interharmonics = tuple(self._params.get("interharmonics_hz", ()))
        self._include_dc = bool(self._params.get("include_dc", True))

        # enforce fundamental included
        if 1 not in self._harmonics:
            self._harmonics = (1,) + tuple(int(h) for h in self._harmonics)

        # sanitize
        if not (0.0 < self._lam <= 1.0):
            self._lam = 0.995
        if self._delta <= 0:
            self._delta = 1e3
        if self._search_step <= 0:
            self._search_step = 0.05
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0
        if self._freq_penalty < 0:
            self._freq_penalty = 0.0
        if self._nom_penalty < 0:
            self._nom_penalty = 0.0
        if self._warmup < 0:
            self._warmup = 0
        if self._p_sym_every < 1:
            self._p_sym_every = 25

        dim = self._param_dim()
        self._theta = np.zeros(dim, dtype=float)
        self._P = self._delta * np.eye(dim, dtype=float)

        self._k = 0
        self._upd_count = 0
        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

    @property
    def latency_samples(self) -> int:
        # strictly online, no explicit window
        return 0

    # -----------------------
    # Core algorithm
    # -----------------------
    def _step(self, v_sample: float) -> float:
        y = float(v_sample)
        if not np.isfinite(y):
            return float(self._f_est)

        k = int(self._k)

        # Warmup: lock frequency to nominal while amplitudes start adapting.
        if k < self._warmup:
            f_use = float(_clamp(self._f_nom, self._min_f, self._max_f))
            phi = self._phi(k, f_use)
            _ = self._rls_update(phi, y)
            self._f_est = f_use
            self._k += 1
            return float(self._f_est)

        f_prev = float(self._f_est)

        # 1) candidate grid around previous estimate
        grid = self._candidate_grid(f_prev)

        # 2) pick best frequency using normalized innovation cost
        best_f = f_prev
        best_phi = None
        best_cost = float("inf")

        for f_c in grid:
            f_c = float(f_c)
            phi = self._phi(k, f_c)

            # prediction
            y_hat = float(phi.T @ self._theta)
            e = float(y - y_hat)

            # innovation variance proxy
            P_phi = self._P @ phi
            S = float(self._lam + float(phi.T @ P_phi))
            if (not np.isfinite(S)) or S <= self.EPS:
                S = self.EPS

            # normalized cost
            cost = float((e * e) / S)

            # smoothness penalty (keep near previous)
            if self._freq_penalty > 0.0:
                df = f_c - f_prev
                cost += float(self._freq_penalty * (df * df))

            # weak nominal prior (prevents drifting to low edge under interharmonics)
            if self._nom_penalty > 0.0:
                dn = f_c - float(self._f_nom)
                cost += float(self._nom_penalty * (dn * dn))

            if cost < best_cost:
                best_cost = cost
                best_f = f_c
                best_phi = phi

        if best_phi is None:
            best_f = f_prev
            best_phi = self._phi(k, best_f)

        # 3) commit frequency
        self._f_est = float(_clamp(best_f, self._min_f, self._max_f))

        # 4) single RLS update with chosen basis
        _ = self._rls_update(best_phi, y)

        self._k += 1
        return float(self._f_est)
