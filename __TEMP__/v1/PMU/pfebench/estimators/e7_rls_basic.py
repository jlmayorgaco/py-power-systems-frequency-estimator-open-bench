#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/estimators/e7_rls_basic.py

E7 — Recursive Least Squares (RLS) basic (classic) frequency estimator.

Key robustness upgrades (still same AR(2)-RLS idea):
- Physical projection: constrain the learned AR(2) to a plausible sinusoidal pole radius r
  (prevents a2 drifting to nonsense and collapsing freq).
- Numerically safer covariance update (Joseph form + symmetrize P).
- Avoid lam=1.0 in tuning grid (optional but recommended; still accepted if user sets it).
- Optional "hold-last" if parameters look invalid.
"""

from __future__ import annotations

from typing import List
import numpy as np

from pfebench.estimators.base import BaseEstimator, TuningParam


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class RecursiveLeastSquaresBasicEstimator(BaseEstimator):
    NAME: str = "RecursiveLeastSquaresBasicEstimator"
    FAMILY: str = "Regression-Temporal"

    NOMINAL_FREQ_HZ: float = 60.0
    DEFAULT_FS: float = 1000.0
    EPS: float = 1e-12

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Keep grid compact. fs comes from scenario, do NOT tune it.
        Note: we intentionally omit lam=1.0 from grid (it harms tracking under steps).
        Users can still set lam=1.0 manually if they want.
        """
        return [
            TuningParam(
                name="lam",
                default=0.995,
                type="float",
                values=[0.98, 0.99, 0.995, 0.998],  # <-- removed 1.0
                description="RLS forgetting factor (lower tracks faster, noisier).",
            ),
            TuningParam(
                name="delta",
                default=1e3,
                type="float",
                values=[1e2, 1e3, 1e4],
                description="Initial covariance scale (bigger adapts faster at start).",
            ),
            TuningParam(
                name="min_freq_hz",
                default=30.0,
                type="float",
                values=[20.0, 30.0, 40.0],
                description="Minimum plausible frequency clamp (Hz).",
            ),
            TuningParam(
                name="max_freq_hz",
                default=70.0,
                type="float",
                values=[70.0, 80.0, 90.0],
                description="Maximum plausible frequency clamp (Hz).",
            ),
            # Physical projection knobs (safe defaults)
            TuningParam(
                name="r_min",
                default=0.92,
                type="float",
                values=[0.90, 0.92, 0.95],
                description="Min pole radius r for sinusoid model projection.",
            ),
            TuningParam(
                name="r_max",
                default=1.02,
                type="float",
                values=[1.00, 1.02, 1.05],
                description="Max pole radius r for sinusoid model projection.",
            ),
            TuningParam(
                name="p_sym_every",
                default=25,
                type="int",
                values=[10, 25, 50],
                description="Every N samples, force P=(P+P^T)/2 for numerical stability.",
            ),
        ]

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

        # Projection / numeric knobs
        self._r_min = float(self._params.get("r_min", 0.92))
        self._r_max = float(self._params.get("r_max", 1.02))
        self._p_sym_every = int(self._params.get("p_sym_every", 25))

        # sanitize
        if not (0.0 < self._lam <= 1.0):
            self._lam = 0.995
        if self._delta <= 0:
            self._delta = 1e3
        if self._min_f <= 0:
            self._min_f = 30.0
        if self._max_f <= self._min_f:
            self._max_f = self._min_f + 1.0

        # sanitize projection bounds
        if not np.isfinite(self._r_min) or self._r_min <= 0.0:
            self._r_min = 0.92
        if not np.isfinite(self._r_max) or self._r_max <= 0.0:
            self._r_max = 1.02
        if self._r_max < self._r_min:
            self._r_max = self._r_min

        if self._p_sym_every < 1:
            self._p_sym_every = 25

        # RLS state
        self._theta = np.zeros(2, dtype=float)  # [a1, a2]
        self._P = self._delta * np.eye(2, dtype=float)

        # sample history
        self._x1 = 0.0
        self._x2 = 0.0
        self._k = 0

        # output state
        self._f_est = float(_clamp(self._f_nom, self._min_f, self._max_f))

        # bookkeeping
        self._upd_count = 0

    @property
    def latency_samples(self) -> int:
        # Needs x[k-1] and x[k-2]
        return 2

    # -----------------------
    # Core algorithm helpers
    # -----------------------
    def _project_theta_sinusoid(self) -> None:
        """
        Project AR(2) parameters (a1,a2) to a plausible sinusoidal model:
          a2 = r^2 with r in [r_min, r_max]
          a1 = -2 r cos(omega)  (we keep cos(omega) implied by current a1/a2, then recompute a1)
        This prevents non-physical a2 drift that produces garbage frequency.
        """
        a1 = float(self._theta[0])
        a2 = float(self._theta[1])

        # If a2 invalid, don't project aggressively—just keep last estimate
        if not np.isfinite(a2) or a2 <= self.EPS:
            return

        r = float(np.sqrt(max(a2, self.EPS)))
        r_proj = float(_clamp(r, self._r_min, self._r_max))
        a2_proj = r_proj * r_proj

        # infer cos from current parameters (using original a1, a2)
        denom = 2.0 * r
        if denom <= self.EPS or not np.isfinite(denom):
            return

        cosw = (-a1) / denom
        cosw = float(_clamp(cosw, -1.0, 1.0))

        # recompute a1 on projected radius
        a1_proj = -2.0 * r_proj * cosw

        self._theta[0] = float(a1_proj)
        self._theta[1] = float(a2_proj)

    def _theta_to_freq(self) -> float:
        a1 = float(self._theta[0])
        a2 = float(self._theta[1])

        if (not np.isfinite(a1)) or (not np.isfinite(a2)):
            return float(self._f_est)

        # Need a2 positive to interpret r^2
        if a2 <= self.EPS:
            return float(self._f_est)

        r = float(np.sqrt(max(a2, self.EPS)))
        denom = 2.0 * r
        if denom <= self.EPS:
            return float(self._f_est)

        c = (-a1) / denom
        c = float(_clamp(c, -1.0, 1.0))

        omega = float(np.arccos(c))  # [0, pi]
        f = omega * self._fs / (2.0 * np.pi)

        # Clamp output for safety
        f = float(_clamp(f, self._min_f, self._max_f))
        return f

    # -----------------------
    # Online step
    # -----------------------
    def _step(self, v_sample: float) -> float:
        xk = float(v_sample)
        if not np.isfinite(xk):
            return float(self._f_est)

        # Warmup: fill buffer
        if self._k < 2:
            self._x2 = self._x1
            self._x1 = xk
            self._k += 1
            return float(self._f_est)

        # Regressor: x[k] ≈ phi^T theta, phi = [-x1, -x2]
        phi = np.array([-self._x1, -self._x2], dtype=float)
        y = xk

        P = self._P
        lam = float(self._lam)

        # Gain
        P_phi = P @ phi
        denom = lam + float(phi.T @ P_phi)
        if (not np.isfinite(denom)) or denom <= self.EPS:
            denom = self.EPS
        K = P_phi / denom  # shape (2,)

        # Error
        y_hat = float(phi.T @ self._theta)
        e = y - y_hat

        # Theta update
        self._theta = self._theta + K * e

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(2, dtype=float)
        KH = np.outer(K, phi)  # (2x2)
        P_new = (I - KH) @ P @ (I - KH).T + np.outer(K, K) * lam  # stable-ish
        # Normalize by lambda like standard RLS
        P_new = P_new / lam
        self._P = P_new

        # occasionally enforce symmetry (keeps P PSD-ish numerically)
        self._upd_count += 1
        if self._p_sym_every > 0 and (self._upd_count % self._p_sym_every == 0):
            self._P = 0.5 * (self._P + self._P.T)

        # shift history
        self._x2 = self._x1
        self._x1 = xk
        self._k += 1

        # Project to physical sinusoid model before converting to f
        self._project_theta_sinusoid()

        # Update estimate
        f_new = self._theta_to_freq()
        if np.isfinite(f_new):
            self._f_est = float(f_new)

        return float(self._f_est)
