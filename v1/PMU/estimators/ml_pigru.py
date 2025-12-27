# estimators/ml_pigru.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Optional, Dict
import json
import os

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
# Minimal, safe fallback estimator (never depends on torch)
# ============================================================

@dataclass(frozen=True)
class _FallbackIpDFTConfig:
    fs_hz: float
    f0_hz: float = 60.0
    cycles: int = 4
    search_hz: float = 6.0
    remove_dc: bool = True


class _FallbackIpDFT:
    """
    Tiny robust baseline: Hann + bounded peak + parabolic interpolation.
    Used only if PI-GRU cannot be loaded.
    """

    def __init__(self, cfg: _FallbackIpDFTConfig):
        self.cfg = cfg
        self.fs_hz = float(cfg.fs_hz)
        self.f0_hz = float(cfg.f0_hz)
        self.cycles = int(cfg.cycles)
        self.search_hz = float(cfg.search_hz)

        self.N = int(round((self.fs_hz / max(self.f0_hz, 1e-9)) * self.cycles))
        self.N = max(self.N, 16)
        self.win = np.hanning(self.N).astype(float)
        self.res_hz = self.fs_hz / self.N

        self.buf = deque(maxlen=self.N)
        self._last_f = float(self.f0_hz)

    def reset(self) -> None:
        self.buf.clear()
        self._last_f = float(self.f0_hz)

    def _bounded_peak_bin(self, sp: np.ndarray) -> int:
        f_center = self._last_f if np.isfinite(self._last_f) else self.f0_hz
        k_center = int(round(f_center / self.res_hz))

        k_lo = max(1, int(round((f_center - self.search_hz) / self.res_hz)))
        k_hi = min(len(sp) - 2, int(round((f_center + self.search_hz) / self.res_hz)))
        if k_hi <= k_lo:
            k_lo = max(1, k_center - 2)
            k_hi = min(len(sp) - 2, k_center + 2)

        return int(k_lo + np.argmax(sp[k_lo : k_hi + 1]))

    def step(self, z: float) -> float:
        self.buf.append(float(z))
        if len(self.buf) < self.N:
            return float(self._last_f)

        x = np.asarray(self.buf, dtype=float)
        if bool(self.cfg.remove_dc):
            x = x - float(np.mean(x))

        sp = np.abs(np.fft.rfft(x * self.win))
        k = self._bounded_peak_bin(sp)

        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        denom = (a - 2.0 * b + c)
        if denom == 0.0:
            delta = 0.0
        else:
            delta = 0.5 * (a - c) / denom
            delta = float(np.clip(delta, -0.5, 0.5))

        self._last_f = float((k + delta) * self.res_hz)
        return float(self._last_f)

    @property
    def latency_samples(self) -> int:
        return int(max(1, 0.5 * self.N))


# ============================================================
# PI-GRU runtime wrapper (tries torch; otherwise fallback)
# ============================================================

class _PIGRURuntime:
    """
    Runtime object with a .step(v)->f API.

    Loading policy:
      1) If torch is available and model_path exists: load torchscript or state_dict-ish.
      2) Else: fallback to a tiny spectral estimator so the pipeline runs.
    """

    def __init__(self, fs_hz: float, f0_hz: float, model_path: str, config_path: str):
        self.fs_hz = float(fs_hz)
        self.f0_hz = float(f0_hz)
        self.model_path = str(model_path)
        self.config_path = str(config_path)

        self.window_len = 0
        self._use_fallback = True
        self._fallback = _FallbackIpDFT(
            _FallbackIpDFTConfig(fs_hz=self.fs_hz, f0_hz=self.f0_hz, cycles=4, search_hz=6.0, remove_dc=True)
        )

        # Try read config to get window_len if present
        cfg = self._try_read_json(self.config_path)
        self.window_len = int(cfg.get("window_len", cfg.get("window", 0)) or 0)

        # Try to load torch model (optional dependency)
        self._try_load_torch(cfg)

    def reset(self) -> None:
        if self._use_fallback:
            self._fallback.reset()
        else:
            # reset buffers for the torch path
            self._buf.clear()
            self._last = float(self.f0_hz)

    def _try_read_json(self, path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _try_load_torch(self, cfg: Dict[str, Any]) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            return

        try:
            import torch  # optional
        except Exception:
            return

        # Decide window length
        wl = int(cfg.get("window_len", cfg.get("window", 0)) or 0)
        if wl <= 0:
            wl = 128  # safe default if config doesn't say
        self.window_len = wl

        # Buffer for causal inference
        self._buf = deque(maxlen=self.window_len)
        self._last = float(self.f0_hz)

        # Try TorchScript first (most robust)
        model = None
        try:
            model = torch.jit.load(self.model_path, map_location="cpu")
            model.eval()
        except Exception:
            # If it's not torchscript, we *could* try to load a state_dict,
            # but without architecture code it’s impossible to reconstruct safely.
            model = None

        if model is None:
            return

        self._torch = torch
        self._model = model
        self._use_fallback = False

        # Optional output clamp
        self.f_min = float(cfg.get("f_min_hz", 40.0))
        self.f_max = float(cfg.get("f_max_hz", 80.0))
        if self.f_max < self.f_min:
            self.f_min, self.f_max = self.f_max, self.f_min

        # Optional input normalization
        self.center = _as_bool(cfg.get("center", True))
        self.scale = _as_bool(cfg.get("scale", True))

    def step(self, v_sample: float) -> float:
        if self._use_fallback:
            return float(self._fallback.step(v_sample))

        self._buf.append(float(v_sample))
        if len(self._buf) < int(self._buf.maxlen):
            return float(self.f0_hz)

        x = np.asarray(self._buf, dtype=np.float32)

        if self.center:
            x = x - float(np.mean(x))
        if self.scale:
            s = float(np.sqrt(np.mean(x * x)) + 1e-12)
            x = x / s

        # Model inference
        with self._torch.no_grad():
            inp = self._torch.from_numpy(x).view(1, -1)  # (1, T)
            y = self._model(inp)

        # Support y as tensor / tuple / list
        if isinstance(y, (tuple, list)):
            y0 = y[0]
        else:
            y0 = y

        f_hat = float(self._torch.as_tensor(y0).reshape(-1)[0].item())
        if not np.isfinite(f_hat):
            f_hat = self._last

        # clamp
        f_hat = float(np.clip(f_hat, self.f_min, self.f_max))
        self._last = f_hat
        return f_hat

    @property
    def latency_samples(self) -> int:
        if self._use_fallback:
            return int(self._fallback.latency_samples)
        return int(max(1, self.window_len))


# ============================================================
# Auto-discoverable wrapper (BaseEstimator)
# ============================================================

class PIGRUEstimator(BaseEstimator):
    """
    PI-GRU estimator wrapper.

    Params expected in registry/config:
      - fs_hz (required)
      - f0_hz (optional)
      - model_path (optional, default "pi_gru_pmu.pt")
      - config_path (optional, default "pi_gru_pmu_config.json")

    Behavior:
      - If torch + model exist and is TorchScript: uses it.
      - Otherwise falls back to a tiny spectral baseline (so the pipeline runs).
    """

    NAME = "PI-GRU"
    FAMILY = "Neural (Physics-Informed)"

    def reset(self) -> None:
        fs_hz = float(self._params["fs_hz"])
        f0_hz = float(self._params.get("f0_hz", 60.0))
        model_path = str(self._params.get("model_path", "pi_gru_pmu.pt"))
        cfg_path = str(self._params.get("config_path", "pi_gru_pmu_config.json"))

        self._impl = _PIGRURuntime(
            fs_hz=fs_hz,
            f0_hz=f0_hz,
            model_path=model_path,
            config_path=cfg_path,
        )
        self._impl.reset()

    def step(self, v_sample: float) -> float:
        return float(self._impl.step(v_sample))

    @property
    def latency_samples(self) -> int:
        return int(getattr(self._impl, "latency_samples", 1))
