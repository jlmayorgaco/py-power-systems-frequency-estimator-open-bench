# estimators/koopman.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional
import numpy as np

from .base import BaseEstimator


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(x)


# ============================================================
# 1) Hankel-DMD (Delay-Embedding Koopman)
# ============================================================

@dataclass(frozen=True)
class HankelDMDConfig:
    fs_hz: float
    f0_hz: float = 60.0
    embed_dim: int = 80
    window_cols: int = 120
    rank: int = 12
    smooth_win: int = 10
    f_search_hz: float = 10.0
    ridge: float = 1e-8


class HankelDMDFrequency:
    def __init__(self, cfg: HankelDMDConfig):
        self.cfg = cfg
        self.dt = 1.0 / float(cfg.fs_hz)

        self.m = int(cfg.embed_dim)
        self.L = int(cfg.window_cols)
        self.r = int(cfg.rank)
        self.smooth_win = int(cfg.smooth_win)

        self.raw_needed = self.m + self.L - 1
        self.buf = deque(maxlen=self.raw_needed)

        self._f_hist = deque(maxlen=max(1, self.smooth_win))
        self._last_f = float(cfg.f0_hz)

    def reset(self) -> None:
        self.buf.clear()
        self._f_hist.clear()
        self._last_f = float(self.cfg.f0_hz)

    def _build_hankel(self, x: np.ndarray) -> np.ndarray:
        m, L = self.m, self.L
        H = np.empty((m, L), dtype=float)
        for j in range(L):
            H[:, j] = x[j : j + m]
        return H

    def _estimate_once(self, x: np.ndarray) -> Optional[float]:
        H = self._build_hankel(x)
        X = H[:, :-1]
        Y = H[:, 1:]

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        if s.size == 0:
            return None

        r = min(self.r, int(s.size))
        s_r = s[:r]
        if not np.all(np.isfinite(s_r)) or float(np.min(s_r)) <= 0.0:
            return None

        U_r = U[:, :r]
        V_r = Vt[:r, :].T

        inv_s = 1.0 / (s_r + float(self.cfg.ridge))
        A_tilde = (U_r.T @ Y @ V_r) * inv_s[None, :]

        eigvals = np.linalg.eigvals(A_tilde)
        freqs = np.angle(eigvals) / (2.0 * np.pi * self.dt)

        f0 = float(self.cfg.f0_hz)
        band = float(self.cfg.f_search_hz)
        mask = (freqs > 0.0) & (freqs >= f0 - band) & (freqs <= f0 + band)
        if not np.any(mask):
            return None

        cand = freqs[mask]
        return float(cand[np.argmin(np.abs(cand - self._last_f))])

    def step(self, v: float) -> float:
        self.buf.append(float(v))
        if len(self.buf) < self.raw_needed:
            return float(self.cfg.f0_hz)

        x = np.asarray(self.buf, dtype=float)
        f_hat = self._estimate_once(x)
        if f_hat is None or not np.isfinite(f_hat):
            f_hat = self._last_f

        self._last_f = float(f_hat)
        self._f_hist.append(float(f_hat))
        return float(np.mean(self._f_hist))


# ============================================================
# 2) Streaming EDMD (Recursive Koopman)
# ============================================================

@dataclass(frozen=True)
class StreamingEDMDConfig:
    fs_hz: float
    f0_hz: float = 60.0

    delays: int = 8
    include_nl: bool = True
    lam: float = 0.995
    delta: float = 1e3
    smooth_win: int = 10

    f_search_hz: float = 10.0
    ridge: float = 1e-8


class StreamingEDMDFrequency:
    def __init__(self, cfg: StreamingEDMDConfig):
        self.cfg = cfg
        self.dt = 1.0 / float(cfg.fs_hz)

        self.d = int(cfg.delays)
        if self.d < 2:
            raise ValueError("StreamingEDMDConfig.delays must be >= 2")

        self.buf = deque(maxlen=self.d)
        self.p = self._feat_dim()

        self.P = (float(cfg.delta) * np.eye(self.p)).astype(float)
        self.K = np.eye(self.p, dtype=float)

        self._f_hist = deque(maxlen=max(1, int(cfg.smooth_win)))
        self._last_f = float(cfg.f0_hz)
        self._prev_phi: Optional[np.ndarray] = None

    def _feat_dim(self) -> int:
        base = self.d
        return base if not self.cfg.include_nl else base + 4

    def reset(self) -> None:
        self.buf.clear()
        self.P[:] = float(self.cfg.delta) * np.eye(self.p)
        self.K[:] = np.eye(self.p)
        self._f_hist.clear()
        self._last_f = float(self.cfg.f0_hz)
        self._prev_phi = None

    def _lift(self, x_delays: np.ndarray) -> np.ndarray:
        if not self.cfg.include_nl:
            return x_delays.astype(float)

        m = float(np.mean(x_delays))
        e = float(np.mean(x_delays * x_delays))
        nl = np.array([m, e, np.tanh(m), np.tanh(e)], dtype=float)
        return np.concatenate([x_delays.astype(float), nl], axis=0)

    def _rls_update(self, phi: np.ndarray, phi_next: np.ndarray) -> None:
        lam = float(self.cfg.lam)

        Pphi = self.P @ phi
        denom = lam + float(phi.T @ Pphi)
        g = Pphi / (denom + 1e-12)

        e = phi_next - (self.K @ phi)

        self.K = self.K + np.outer(e, g)
        self.P = (self.P - np.outer(g, phi.T @ self.P)) / (lam + 1e-12)

        self.P = 0.5 * (self.P + self.P.T)
        self.P.flat[:: self.p + 1] += float(self.cfg.ridge)

    def _freq_from_K(self) -> Optional[float]:
        eigvals = np.linalg.eigvals(self.K)
        freqs = np.angle(eigvals) / (2.0 * np.pi * self.dt)

        f0 = float(self.cfg.f0_hz)
        band = float(self.cfg.f_search_hz)
        mask = (freqs > 0.0) & (freqs >= f0 - band) & (freqs <= f0 + band)
        if not np.any(mask):
            return None

        cand = freqs[mask]
        return float(cand[np.argmin(np.abs(cand - self._last_f))])

    def step(self, v: float) -> float:
        self.buf.append(float(v))
        if len(self.buf) < self.d:
            return float(self.cfg.f0_hz)

        xk = np.asarray(self.buf, dtype=float)
        phi = self._lift(xk)

        if self._prev_phi is None:
            self._prev_phi = phi
            return float(self.cfg.f0_hz)

        self._rls_update(self._prev_phi, phi)
        self._prev_phi = phi

        f_hat = self._freq_from_K()
        if f_hat is None or not np.isfinite(f_hat):
            f_hat = self._last_f

        self._last_f = float(f_hat)
        self._f_hist.append(float(f_hat))
        return float(np.mean(self._f_hist))


# ============================================================
# 3) Windowed Koopman (RKDPmu-style, self-contained)
#    = EDMD on a sliding window + spectral extraction
# ============================================================

@dataclass(frozen=True)
class WindowedKoopmanConfig:
    fs_hz: float
    f0_hz: float = 60.0

    window_samples: int = 400      # sliding window length
    delays: int = 20               # lifting with delays inside window
    rank: int = 16                 # truncated SVD rank on lifted snapshots
    smooth_win: int = 10

    f_search_hz: float = 10.0
    ridge: float = 1e-8
    center: bool = True            # remove mean in window


class WindowedKoopmanFrequency:
    """
    Self-contained "RKDPmu-style" Koopman estimator:

    - Maintain a raw sliding window of length W
    - Build delay-lifted snapshots inside that window
    - Estimate Koopman operator (DMD/EDMD) with truncated SVD
    - Extract frequency from eigenvalue angle
    """

    def __init__(self, cfg: WindowedKoopmanConfig):
        self.cfg = cfg
        self.dt = 1.0 / float(cfg.fs_hz)

        self.W = int(cfg.window_samples)
        self.d = int(cfg.delays)
        if self.d < 2:
            raise ValueError("WindowedKoopmanConfig.delays must be >= 2")
        if self.W <= self.d + 2:
            raise ValueError("window_samples must be sufficiently larger than delays")

        self.r = int(cfg.rank)
        self.buf = deque(maxlen=self.W)

        self._f_hist = deque(maxlen=max(1, int(cfg.smooth_win)))
        self._last_f = float(cfg.f0_hz)

    def reset(self) -> None:
        self.buf.clear()
        self._f_hist.clear()
        self._last_f = float(self.cfg.f0_hz)

    def _make_delay_matrix(self, x: np.ndarray) -> np.ndarray:
        # Create snapshots of delays:
        # Phi_k = [x[k], x[k-1], ..., x[k-d+1]] for k = d-1..W-1
        W, d = x.size, self.d
        cols = W - d + 1
        Phi = np.empty((d, cols), dtype=float)
        for j in range(cols):
            seg = x[j : j + d]
            Phi[:, j] = seg[::-1]
        return Phi

    def _estimate(self, x: np.ndarray) -> Optional[float]:
        if self.cfg.center:
            x = x - float(np.mean(x))

        Phi = self._make_delay_matrix(x)  # (d, cols)
        # DMD on lifted snapshots
        X = Phi[:, :-1]
        Y = Phi[:, 1:]

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        if s.size == 0:
            return None

        r = min(self.r, int(s.size))
        s_r = s[:r]
        if not np.all(np.isfinite(s_r)) or float(np.min(s_r)) <= 0.0:
            return None

        U_r = U[:, :r]
        V_r = Vt[:r, :].T

        inv_s = 1.0 / (s_r + float(self.cfg.ridge))
        A_tilde = (U_r.T @ Y @ V_r) * inv_s[None, :]

        eigvals = np.linalg.eigvals(A_tilde)
        freqs = np.angle(eigvals) / (2.0 * np.pi * self.dt)

        f0 = float(self.cfg.f0_hz)
        band = float(self.cfg.f_search_hz)
        mask = (freqs > 0.0) & (freqs >= f0 - band) & (freqs <= f0 + band)
        if not np.any(mask):
            return None

        cand = freqs[mask]
        return float(cand[np.argmin(np.abs(cand - self._last_f))])

    def step(self, v: float) -> float:
        self.buf.append(float(v))
        if len(self.buf) < self.W:
            return float(self.cfg.f0_hz)

        x = np.asarray(self.buf, dtype=float)
        f_hat = self._estimate(x)
        if f_hat is None or not np.isfinite(f_hat):
            f_hat = self._last_f

        self._last_f = float(f_hat)
        self._f_hist.append(float(f_hat))
        return float(np.mean(self._f_hist))


# ============================================================
# Wrappers (BaseEstimator)
# ============================================================

class HankelDMDEstimator(BaseEstimator):
    NAME = "Koopman-HankelDMD"
    FAMILY = "Koopman"

    def reset(self) -> None:
        cfg = HankelDMDConfig(
            fs_hz=float(self._params["fs_hz"]),
            f0_hz=float(self._params.get("f0_hz", 60.0)),
            embed_dim=int(self._params.get("embed_dim", 80)),
            window_cols=int(self._params.get("window_cols", 120)),
            rank=int(self._params.get("rank", 12)),
            smooth_win=int(self._params.get("smooth_win", 10)),
            f_search_hz=float(self._params.get("f_search_hz", 10.0)),
            ridge=float(self._params.get("ridge", 1e-8)),
        )
        self._impl = HankelDMDFrequency(cfg)

    def step(self, v_sample: float) -> float:
        return float(self._impl.step(v_sample))

    @property
    def latency_samples(self) -> int:
        m = int(self._params.get("embed_dim", 80))
        L = int(self._params.get("window_cols", 120))
        return int(0.5 * (m + L - 1))


class StreamingEDMDEstimator(BaseEstimator):
    NAME = "Koopman-StreamingEDMD"
    FAMILY = "Koopman"

    def reset(self) -> None:
        cfg = StreamingEDMDConfig(
            fs_hz=float(self._params["fs_hz"]),
            f0_hz=float(self._params.get("f0_hz", 60.0)),
            delays=int(self._params.get("delays", 8)),
            include_nl=_as_bool(self._params.get("include_nl", True)),
            lam=float(self._params.get("lam", 0.995)),
            delta=float(self._params.get("delta", 1e3)),
            smooth_win=int(self._params.get("smooth_win", 10)),
            f_search_hz=float(self._params.get("f_search_hz", 10.0)),
            ridge=float(self._params.get("ridge", 1e-8)),
        )
        self._impl = StreamingEDMDFrequency(cfg)

    def step(self, v_sample: float) -> float:
        return float(self._impl.step(v_sample))

    @property
    def latency_samples(self) -> int:
        return int(max(1, int(self._params.get("delays", 8)) - 1))


class WindowedKoopmanEstimator(BaseEstimator):
    NAME = "Koopman-RKDPmu"
    FAMILY = "Koopman"

    def reset(self) -> None:
        cfg = WindowedKoopmanConfig(
            fs_hz=float(self._params["fs_hz"]),
            f0_hz=float(self._params.get("f0_hz", 60.0)),
            window_samples=int(self._params.get("window_samples", 400)),
            delays=int(self._params.get("delays", 20)),
            rank=int(self._params.get("rank", 16)),
            smooth_win=int(self._params.get("smooth_win", 10)),
            f_search_hz=float(self._params.get("f_search_hz", 10.0)),
            ridge=float(self._params.get("ridge", 1e-8)),
            center=_as_bool(self._params.get("center", True)),
        )
        self._impl = WindowedKoopmanFrequency(cfg)

    def step(self, v_sample: float) -> float:
        return float(self._impl.step(v_sample))

    @property
    def latency_samples(self) -> int:
        W = int(self._params.get("window_samples", 400))
        return int(0.5 * W)
