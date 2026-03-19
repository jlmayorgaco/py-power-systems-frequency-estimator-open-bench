# estimators/ml_classic.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any
import os
import pickle

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
# Feature engineering (Q1-reasonable, causal, low-cost)
# ============================================================


@dataclass(frozen=True)
class ClassicMLFeaturesConfig:
    window_len: int = 80
    use_raw: bool = True  # raw window samples
    use_diff: bool = True  # first differences
    use_stats: bool = True  # mean, std, rms, skew-ish, kurt-ish
    use_zcr: bool = True  # zero-crossing rate
    clip_value: float = 5.0  # safety clip for extreme outliers


class ClassicFeatureExtractor:
    """
    Causal features from a 1D signal window (v[k-N+1:k]).
    Goal: robust, low-cost, explainable.
    """

    def __init__(self, cfg: ClassicMLFeaturesConfig):
        self.cfg = cfg

    def transform(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.size != int(self.cfg.window_len):
            raise ValueError(f"Expected window_len={self.cfg.window_len}, got {w.size}")

        # Robust clip (protects classical ML against spikes)
        cv = float(self.cfg.clip_value)
        if cv > 0.0:
            w = np.clip(w, -cv, cv)

        feats = []

        # 1) Raw samples (normalized by RMS to help generalization)
        if bool(self.cfg.use_raw):
            rms = float(np.sqrt(np.mean(w * w)) + 1e-12)
            feats.append((w / rms).astype(float))

        # 2) First differences (causal)
        if bool(self.cfg.use_diff):
            dw = np.diff(w, prepend=w[0])
            rmsd = float(np.sqrt(np.mean(dw * dw)) + 1e-12)
            feats.append((dw / rmsd).astype(float))

        # 3) Simple statistics
        if bool(self.cfg.use_stats):
            m = float(np.mean(w))
            s = float(np.std(w) + 1e-12)
            rms = float(np.sqrt(np.mean(w * w)) + 1e-12)

            # “skew-ish” and “kurt-ish” (stable)
            z = (w - m) / s
            skew = float(np.mean(z**3))
            kurt = float(np.mean(z**4))

            p2p = float(np.max(w) - np.min(w))
            med = float(np.median(w))
            mad = float(np.median(np.abs(w - med)) + 1e-12)

            feats.append(np.array([m, s, rms, skew, kurt, p2p, med, mad], dtype=float))

        # 4) Zero-crossing rate (ZCR)
        if bool(self.cfg.use_zcr):
            sgn = np.sign(w)
            sgn[sgn == 0.0] = 1.0
            zc = float(np.mean(sgn[1:] != sgn[:-1])) if w.size > 1 else 0.0
            feats.append(np.array([zc], dtype=float))

        if not feats:
            # Always return something deterministic
            return np.zeros((1,), dtype=float)

        return np.concatenate(feats, axis=0)


# ============================================================
# Model loading (sklearn models serialized with pickle/joblib)
# ============================================================


def _load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_sklearn(model: Any, x: np.ndarray) -> float:
    """
    Supports sklearn-like API:
      y = model.predict(X) with X shape (1, d).
    """
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not implement .predict(X).")
    y = model.predict(x.reshape(1, -1))
    return float(np.asarray(y).reshape(-1)[0])


# ============================================================
# Base class: Classic ML windowed regressor
# ============================================================


class _ClassicMLWindowed(BaseEstimator):
    """
    Loads a pretrained classical ML regressor and runs causal inference:
      input: v_sample (float)
      output: f_hat (float, Hz)

    NOTE: This is “classic ML inference”; training is offline.
    """

    FAMILY = "ML-Classic"
    NAME = "ML-Classic-Base"

    def reset(self) -> None:
        # Required params
        self.fs_hz = float(self._params["fs_hz"])
        self.f0_hz = float(self._params.get("f0_hz", 60.0))

        window_len = int(self._params.get("window_len", 80))
        window_len = max(4, window_len)  # safety
        self._win = deque(maxlen=window_len)

        self._feat = ClassicFeatureExtractor(
            ClassicMLFeaturesConfig(
                window_len=window_len,
                use_raw=_as_bool(self._params.get("use_raw", True)),
                use_diff=_as_bool(self._params.get("use_diff", True)),
                use_stats=_as_bool(self._params.get("use_stats", True)),
                use_zcr=_as_bool(self._params.get("use_zcr", True)),
                clip_value=float(self._params.get("clip_value", 5.0)),
            )
        )

        # Model path (required)
        self.model_path = str(self._params["model_path"])
        self._model = _load_pickle(self.model_path)

        # Optional safety clamp output
        self.f_min = float(self._params.get("f_min_hz", 40.0))
        self.f_max = float(self._params.get("f_max_hz", 80.0))
        if self.f_max < self.f_min:
            self.f_min, self.f_max = self.f_max, self.f_min

        self._last = float(self.f0_hz)

    def step(self, v_sample: float) -> float:
        self._win.append(float(v_sample))
        if len(self._win) < int(self._win.maxlen):
            return float(self.f0_hz)

        w = np.asarray(self._win, dtype=float)
        x = self._feat.transform(w)

        try:
            f_hat = _predict_sklearn(self._model, x)
        except Exception:
            f_hat = self._last

        if not np.isfinite(f_hat):
            f_hat = self._last

        f_hat = float(np.clip(f_hat, self.f_min, self.f_max))
        self._last = f_hat
        return f_hat

    @property
    def latency_samples(self) -> int:
        # Structural latency ~ half window
        return int(max(1, 0.5 * int(self._params.get("window_len", 80))))


# ============================================================
# Concrete estimators (auto-discovery via BaseEstimator subclass)
# ============================================================


class SVRFrequencyEstimator(_ClassicMLWindowed):
    """
    Classic baseline: Support Vector Regression (SVR).
    Train offline with sklearn.svm.SVR; serialize with pickle/joblib.
    """

    NAME = "SVR"
    FAMILY = "ML-Classic"


class RandomForestFrequencyEstimator(_ClassicMLWindowed):
    """
    Classic baseline: RandomForestRegressor.
    Train offline with sklearn.ensemble.RandomForestRegressor.
    """

    NAME = "RandomForest"
    FAMILY = "ML-Classic"
