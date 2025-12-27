# estimators/zcd/distributed.py
# ---------------------------------------------------------------------
# Minimal distributed ZCD:
# - One ZCD core per node (you provide per-node scalar samples each step).
# - Optional single-step consensus fusion using Metropolis weights.
#
# This file is framework-agnostic: it returns a dict so you can plug it
# into your messaging / PMU pipeline easily. If you already have a
# DistributedPMU_Output class, adapt the return section accordingly.
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from estimators.zcd.core import ZCDConfig, ZCDEstimatorBase


def metropolis_weights(adj: Mapping[str, list[str]]) -> dict[tuple[str, str], float]:
    """
    Build Metropolis-Hastings weights w_ij for i with neighbors N(i).
    w_ij = 1 / (1 + max(deg(i), deg(j))) for j in N(i); w_ii = 1 - sum_j w_ij
    """
    deg: dict[str, int] = {i: max(1, len(neigh)) for i, neigh in adj.items()}
    W: dict[tuple[str, str], float] = {}
    for i, neigh in adj.items():
        s = 0.0
        for j in neigh:
            wij = 1.0 / (1.0 + float(max(deg[i], deg.get(j, 1))))
            W[(i, j)] = wij
            s += wij
        W[(i, i)] = max(0.0, 1.0 - s)
    return W


class DistributedZCD:
    """
    Distributed ZCD manager.
    Config dict (keys):
      - nodes: list[str]                          (required)
      - adjacency: dict[str, list[str]]           (required for consensus)
      - epsilon, nominal_hz, mode                 (optional, passed to ZCD cores)
      - fuse: "consensus" | "mean" | "none"       (default "consensus")
      - consensus_alpha: float in (0,1]           (default 1.0, single MH step)
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.nodes: list[str] = list(config.get("nodes", []))
        if not self.nodes:
            raise ValueError("DistributedZCD requires config['nodes']")

        eps = float(config.get("epsilon", 0.0))
        nominal = float(config.get("nominal_hz", 60.0))
        mode = str(config.get("mode", "neg_to_pos"))

        self.cores: dict[str, ZCDEstimatorBase] = {
            n: ZCDEstimatorBase(ZCDConfig(epsilon=eps, nominal_hz=nominal, mode=mode))
            for n in self.nodes
        }

        self.adj: dict[str, list[str]] = {
            n: list(config.get("adjacency", {}).get(n, [])) for n in self.nodes
        }
        self.W: dict[tuple[str, str], float] = (
            metropolis_weights(self.adj) if any(self.adj.values()) else {}
        )
        self.fuse_mode: str = str(config.get("fuse", "consensus"))
        self.alpha: float = float(config.get("consensus_alpha", 1.0))  # 1 MH step by default

        # last fused estimates (optional cache)
        self.last_fused_freq: float | None = None
        self.last_fused_rocof: float | None = None

    def reset(self) -> None:
        for core in self.cores.values():
            core.reset()
        self.last_fused_freq = None
        self.last_fused_rocof = None

    def _fuse_consensus(self, local_f: Mapping[str, float]) -> dict[str, float]:
        if not self.W:
            # No graph → fall back to mean
            m = sum(local_f.values()) / max(1, len(local_f))
            return {n: float(m) for n in local_f}

        new_f: dict[str, float] = {}
        for i in local_f:
            acc = self.W.get((i, i), 0.0) * local_f[i]
            for j in self.adj.get(i, []):
                acc += self.W.get((i, j), 0.0) * local_f.get(j, local_f[i])
            # one step (or blend via alpha)
            new_f[i] = float((1.0 - self.alpha) * local_f[i] + self.alpha * acc)
        return new_f

    def step(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """
        Update with one distributed sample.

        sample schema (minimal):
            {
              "timestamp": <float seconds>,
              "nodes": {
                 "bus1": {"value": <float>},
                 "bus2": {"value": <float>},
                 ...
              }
            }

        Returns dict:
            {
              "timestamp": ts,
              "local": { node: {"freq_hz": f, "rocof_hz_s": r} },
              "fused": { "mean_freq_hz": ..., "mean_rocof_hz_s": ...,
                         "consensus_freq_hz": {node: fi, ...} (if consensus) }
            }
        """
        ts = float(sample["timestamp"])
        nodes_payload = sample["nodes"]  # Mapping[str, Mapping[str, Any]]

        # 1) local ZCD updates
        local_freq: dict[str, float] = {}
        local_rocof: dict[str, float] = {}
        for n, core in self.cores.items():
            v = float(nodes_payload.get(n, {}).get("value", 0.0))
            f, r, _crossed, _tc = core.update_scalar(v, ts)
            local_freq[n] = f
            local_rocof[n] = r

        # 2) fusion
        mean_f = sum(local_freq.values()) / max(1, len(local_freq))
        mean_r = sum(local_rocof.values()) / max(1, len(local_rocof))

        fused: dict[str, Any] = {
            "mean_freq_hz": float(mean_f),
            "mean_rocof_hz_s": float(mean_r),
        }

        if self.fuse_mode == "consensus":
            fused_freq = self._fuse_consensus(local_freq)
            fused["consensus_freq_hz"] = fused_freq
            # store an overall average as headline
            self.last_fused_freq = sum(fused_freq.values()) / max(1, len(fused_freq))
            self.last_fused_rocof = mean_r  # RoCoF fusion kept as mean for simplicity
        else:
            self.last_fused_freq = mean_f
            self.last_fused_rocof = mean_r

        return {
            "timestamp": ts,
            "local": {
                n: {"freq_hz": local_freq[n], "rocof_hz_s": local_rocof[n]} for n in self.nodes
            },
            "fused": fused,
        }
