# estimators/ml_sota.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Dict
import os
import json
import numpy as np

from .base import BaseEstimator

# ============================================================
# Torch is optional at import-time (so discovery doesn't break)
# ============================================================

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for ML-SOTA estimators (GRU/LSTM). "
            "Install torch and retry."
        )


# ============================================================
# Models (causal window -> f_hat)
#   IMPORTANT: must not crash module import if torch missing
# ============================================================

if nn is not None:

    class _RNNRegressor(nn.Module):
        def __init__(
            self,
            cell: str,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            cell_l = cell.lower().strip()

            if cell_l == "gru":
                self.rnn = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
            elif cell_l == "lstm":
                self.rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
            else:
                raise ValueError("cell must be 'gru' or 'lstm'")

            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.cell = cell_l

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, input_dim)
            y, _ = self.rnn(x)
            hT = y[:, -1, :]  # last step (causal)
            out = self.head(hT)
            return out  # (B,1)

else:
    # Safe stub so module imports even without torch
    _RNNRegressor = None  # type: ignore


# ============================================================
# Config + loader
# ============================================================


@dataclass(frozen=True)
class RNNConfig:
    cell: str = "gru"  # "gru" or "lstm"
    window_len: int = 80
    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0

    # preprocessing
    normalize: str = "rms"  # "none" | "rms" | "zscore"
    clip_value: float = 5.0

    # safety
    f_min_hz: float = 40.0
    f_max_hz: float = 80.0


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _to_rnn_cfg(d: Dict[str, Any]) -> RNNConfig:
    return RNNConfig(
        cell=str(d.get("cell", "gru")),
        window_len=int(d.get("window_len", 80)),
        input_dim=int(d.get("input_dim", 1)),
        hidden_dim=int(d.get("hidden_dim", 64)),
        num_layers=int(d.get("num_layers", 2)),
        dropout=float(d.get("dropout", 0.0)),
        normalize=str(d.get("normalize", "rms")),
        clip_value=float(d.get("clip_value", 5.0)),
        f_min_hz=float(d.get("f_min_hz", 40.0)),
        f_max_hz=float(d.get("f_max_hz", 80.0)),
    )


# ============================================================
# Preprocess (causal)
# ============================================================


def _preprocess_window(w: np.ndarray, cfg: RNNConfig) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != cfg.window_len:
        raise ValueError(f"Expected window_len={cfg.window_len}, got {w.size}")

    cv = float(cfg.clip_value)
    if cv > 0:
        w = np.clip(w, -cv, cv)

    mode = cfg.normalize.lower().strip()
    if mode == "none":
        return w
    if mode == "rms":
        s = float(np.sqrt(np.mean(w * w)) + 1e-12)
        return w / s
    if mode == "zscore":
        m = float(np.mean(w))
        s = float(np.std(w) + 1e-12)
        return (w - m) / s

    raise ValueError("normalize must be one of: none, rms, zscore")


# ============================================================
# BaseEstimator wrapper for pretrained RNN
# ============================================================


class _RNNEstimator(BaseEstimator):
    FAMILY = "ML-SOTA"
    NAME = "RNN-Base"

    def reset(self) -> None:
        _require_torch()
        if _RNNRegressor is None:
            # extra guard
            raise ImportError("Torch is required but RNN class is unavailable.")

        self.fs_hz = float(self._params["fs_hz"])
        self.f0_hz = float(self._params.get("f0_hz", 60.0))

        # model + cfg paths
        self.model_path = str(self._params["model_path"])
        self.config_path = str(self._params["config_path"])

        cfg_dict = _load_json(self.config_path)
        self.cfg = _to_rnn_cfg(cfg_dict)

        # window buffer
        self._win = deque(maxlen=int(self.cfg.window_len))

        # device
        dev = str(self._params.get("device", "cpu")).lower().strip()
        self.device = torch.device(
            "cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        # build model
        self._net = _RNNRegressor(
            cell=self.cfg.cell,
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)
        self._net.eval()

        # load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        state = torch.load(self.model_path, map_location=self.device)
        # allow either {"state_dict": ...} or raw state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._net.load_state_dict(state, strict=True)

        # safety clamps
        self.f_min = float(self.cfg.f_min_hz)
        self.f_max = float(self.cfg.f_max_hz)
        self._last = float(self.f0_hz)

    def step(self, v_sample: float) -> float:
        self._win.append(float(v_sample))
        if len(self._win) < self._win.maxlen:
            return float(getattr(self, "f0_hz", self._params.get("f0_hz", 60.0)))

        w = np.asarray(self._win, dtype=float)
        w = _preprocess_window(w, self.cfg)

        # shape: (B=1, T, input_dim=1)
        x = torch.tensor(w, dtype=torch.float32, device=self.device).view(1, -1, 1)

        with torch.no_grad():
            y = self._net(x).view(-1).detach().cpu().numpy()

        f_hat = float(y[0]) if y.size else self._last
        if not np.isfinite(f_hat):
            f_hat = self._last

        f_hat = float(np.clip(f_hat, self.f_min, self.f_max))
        self._last = f_hat
        return f_hat

    @property
    def latency_samples(self) -> int:
        # safe even if called before reset()
        w = int(self._params.get("window_len", 80))
        try:
            w = int(getattr(self, "cfg").window_len)  # type: ignore[attr-defined]
        except Exception:
            pass
        return int(0.5 * max(1, w))


# ============================================================
# Concrete estimators (auto-discover via BaseEstimator + NAME)
# ============================================================


class GRUFrequencyEstimator(_RNNEstimator):
    """SOTA baseline: GRU regressor (causal, windowed)."""

    NAME = "GRU"
    FAMILY = "ML-SOTA"


class LSTMFrequencyEstimator(_RNNEstimator):
    """SOTA baseline: LSTM regressor (causal, windowed)."""

    NAME = "LSTM"
    FAMILY = "ML-SOTA"
