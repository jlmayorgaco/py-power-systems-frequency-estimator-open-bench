"""
openfreqbench/core/checkpoint.py

CheckpointManager — interruptible / resumable benchmark execution state.

Each (estimator_id, scenario_id) pair is an atomic work unit.
Results are written atomically immediately after completion.
On restart, already-completed pairs are skipped automatically.

Checkpoint file: <output_dir>/.ofb_checkpoint.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_key(method: str, scenario: str) -> str:
    return f"{method}::{scenario}"


def _config_hash(config: Any) -> str:
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _format_elapsed(started_at_iso: str) -> str:
    try:
        start = datetime.fromisoformat(started_at_iso)
        now   = datetime.now(timezone.utc)
        delta = now - start.replace(tzinfo=timezone.utc) if start.tzinfo is None else now - start
        total_s = int(delta.total_seconds())
        h, rem  = divmod(total_s, 3600)
        m, s    = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
    except Exception:
        return "unknown"


# ── CheckpointManager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages incremental benchmark execution state.

    Checkpoint file layout
    ──────────────────────
    {
      "schema_version": "1.0",
      "session_id": "<uuid4>",
      "started_at": "<ISO>",
      "last_updated": "<ISO>",
      "config_hash": "<sha256[:16]>",
      "completed": {
        "EKF_Freq::G1_E1_Pure_60Hz": {
          "completed_at": "<ISO>",
          "n_mc": 30,
          "result_file": "path/to/report.json"
        }
      },
      "in_progress": null,
      "failed": {
        "RDFT::IBR_Nightmare": {
          "failed_at": "<ISO>",
          "error": "..."
        }
      }
    }

    Atomic write protocol
    ─────────────────────
    Write to ``<file>.tmp`` then ``os.replace()`` to prevent partial files.
    """

    FILENAME = ".ofb_checkpoint.json"

    def __init__(self, output_dir: Path) -> None:
        self._path = Path(output_dir) / self.FILENAME
        self._state: Optional[Dict[str, Any]] = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk; returns None if it does not exist."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                self._state = json.load(fh)
            return self._state
        except (json.JSONDecodeError, OSError):
            return None

    def _save(self) -> None:
        """Atomic write: write to .tmp then rename."""
        if self._state is None:
            return
        self._state["last_updated"] = _now_iso()
        tmp = self._path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        os.replace(tmp, self._path)

    def delete(self) -> None:
        """Delete the checkpoint file (for full restart)."""
        if self._path.exists():
            self._path.unlink()
        self._state = None

    # ── Session initialisation ────────────────────────────────────────────────

    def create(self, config: Any) -> None:
        """Create a fresh checkpoint (overwrites any existing state)."""
        self._state = {
            "schema_version": "1.0",
            "session_id":     str(uuid.uuid4()),
            "started_at":     _now_iso(),
            "last_updated":   _now_iso(),
            "config_hash":    _config_hash(config),
            "completed":      {},
            "in_progress":    None,
            "failed":         {},
        }
        self._save()

    # ── Pair lifecycle ────────────────────────────────────────────────────────

    def mark_started(self, method: str, scenario: str) -> None:
        """Call before starting a (method, scenario) pair."""
        if self._state is None:
            return
        self._state["in_progress"] = {
            "pair":       _pair_key(method, scenario),
            "started_at": _now_iso(),
        }
        self._save()

    def mark_completed(
        self,
        method: str,
        scenario: str,
        result_file: str = "",
        n_mc: int = 0,
    ) -> None:
        """Call immediately after a pair completes. Writes to disk atomically."""
        if self._state is None:
            return
        key = _pair_key(method, scenario)
        self._state["completed"][key] = {
            "completed_at": _now_iso(),
            "n_mc":         n_mc,
            "result_file":  result_file,
        }
        self._state["in_progress"] = None
        # Remove from failed if it was previously failing
        self._state["failed"].pop(key, None)
        self._save()

    def mark_failed(self, method: str, scenario: str, error_msg: str) -> None:
        """Call if a pair raises an exception."""
        if self._state is None:
            return
        key = _pair_key(method, scenario)
        self._state["failed"][key] = {
            "failed_at": _now_iso(),
            "error":     str(error_msg)[:500],
        }
        self._state["in_progress"] = None
        self._save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_completed(self, method: str, scenario: str) -> bool:
        """Returns True if this pair is already done."""
        if self._state is None:
            return False
        return _pair_key(method, scenario) in self._state.get("completed", {})

    def is_failed(self, method: str, scenario: str) -> bool:
        """Returns True if this pair previously failed."""
        if self._state is None:
            return False
        return _pair_key(method, scenario) in self._state.get("failed", {})

    def should_run(
        self,
        method: str,
        scenario: str,
        mode: str,   # "resume" | "restart" | "partial"
    ) -> bool:
        """
        Returns False if this pair should be skipped.

        resume  → skip completed pairs
        partial → skip completed, re-run failed
        restart → run all pairs (no skipping)
        """
        if mode == "restart":
            return True
        if mode in ("resume", "partial"):
            return not self.is_completed(method, scenario)
        return True

    # ── Counts ────────────────────────────────────────────────────────────────

    def count_completed(self) -> int:
        if self._state is None:
            return 0
        return len(self._state.get("completed", {}))

    def count_failed(self) -> int:
        if self._state is None:
            return 0
        return len(self._state.get("failed", {}))

    def list_completed_pairs(self) -> List[str]:
        if self._state is None:
            return []
        return sorted(self._state.get("completed", {}).keys())

    def list_failed_pairs(self) -> List[Tuple[str, str]]:
        """Returns list of (pair_key, error_msg)."""
        if self._state is None:
            return []
        return [
            (k, v.get("error", ""))
            for k, v in self._state.get("failed", {}).items()
        ]

    @property
    def session_id(self) -> str:
        if self._state is None:
            return "none"
        return self._state.get("session_id", "none")

    @property
    def started_at(self) -> str:
        if self._state is None:
            return ""
        return self._state.get("started_at", "")

    @property
    def elapsed(self) -> str:
        return _format_elapsed(self.started_at) if self.started_at else "n/a"

    @property
    def in_progress(self) -> Optional[Dict[str, str]]:
        if self._state is None:
            return None
        return self._state.get("in_progress")

    # ── Interactive startup dialog ────────────────────────────────────────────

    def startup_dialog(
        self,
        config: Any,
        n_total: int,
        *,
        auto_resume: bool = False,
        auto_restart: bool = False,
    ) -> str:
        """
        Show the interactive checkpoint dialog and return the chosen mode.

        Returns one of: "resume" | "restart" | "partial"

        Parameters
        ----------
        config:
            The run configuration object (used for config hash comparison).
        n_total:
            Total number of (method × scenario) pairs in this run.
        auto_resume:
            If True, skip dialog and automatically resume.
        auto_restart:
            If True, skip dialog and automatically restart (deletes checkpoint).
        """
        checkpoint = self.load()

        if checkpoint is None:
            print("No checkpoint found. Starting fresh run.")
            self.create(config)
            return "restart"

        # Check config hash — warn if config changed
        new_hash = _config_hash(config)
        old_hash = checkpoint.get("config_hash", "")
        config_changed = old_hash and new_hash != old_hash

        n_done   = self.count_completed()
        n_failed = self.count_failed()

        SEP = "=" * 60
        print(f"\n{SEP}")
        print(f"  CHECKPOINT FOUND — Session: {self.session_id[:8]}...")
        print(f"  Started     : {self.started_at}")
        print(f"  Elapsed     : {self.elapsed}")
        print(f"  Progress    : {n_done}/{n_total} pairs completed")
        if n_failed:
            print(f"  Failed pairs: {self.list_failed_pairs()!r}")
        if config_changed:
            print("  [WARNING] Run config has changed since checkpoint was created.")
        print(SEP)

        # Non-interactive auto modes
        if auto_resume:
            print(f"  Auto-resuming (--resume). Skipping {n_done} completed pairs.")
            return "resume"
        if auto_restart:
            print("  Auto-restarting (--restart). Deleting checkpoint.")
            self.delete()
            self.create(config)
            return "restart"

        print("\nOptions:")
        print("  [R] Resume   — skip completed pairs (default)")
        print("  [S] Restart  — delete checkpoint, start from scratch")
        print("  [P] Partial  — re-run only failed pairs, skip completed")
        print("  [L] List     — list completed pairs and exit")

        try:
            choice = input("\nChoice [R/S/P/L]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "R"

        if choice == "L":
            print("\nCompleted pairs:")
            for pair in self.list_completed_pairs():
                print(f"  ✓ {pair}")
            sys.exit(0)

        if choice == "S":
            self.delete()
            self.create(config)
            return "restart"

        if choice == "P":
            print(f"Partial restart. Skipping {n_done} completed, "
                  f"re-running {n_failed} failed pair(s).")
            return "partial"

        # Default: R or anything else → resume
        print(f"Resuming. Skipping {n_done} completed pair(s).")
        return "resume"
