"""
tests/unit/test_checkpoint.py

Unit tests for CheckpointManager (BLOCK 0).

All tests use tmp_path so nothing is written to the real results directory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from openfreqbench.core.checkpoint import CheckpointManager, _pair_key


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mgr(tmp_path):
    return CheckpointManager(tmp_path)


CONFIG = {"methods": ["EKF_Freq"], "scenarios": ["G1_E1"], "n_runs": 10}


# ── Creation and persistence ──────────────────────────────────────────────────

def test_load_returns_none_when_no_file(mgr):
    assert mgr.load() is None


def test_create_writes_file(mgr, tmp_path):
    mgr.create(CONFIG)
    assert (tmp_path / ".ofb_checkpoint.json").exists()


def test_load_after_create(mgr):
    mgr.create(CONFIG)
    state = mgr.load()
    assert state is not None
    assert "session_id" in state
    assert "started_at" in state
    assert state["completed"] == {}
    assert state["failed"] == {}


def test_session_id_is_uuid(mgr):
    mgr.create(CONFIG)
    import uuid
    uuid.UUID(mgr.session_id)  # raises if invalid


def test_two_creates_have_different_session_ids(tmp_path):
    m1 = CheckpointManager(tmp_path)
    m2 = CheckpointManager(tmp_path)
    m1.create(CONFIG)
    sid1 = m1.session_id
    m2.create(CONFIG)
    sid2 = m2.session_id
    assert sid1 != sid2


# ── Pair lifecycle ────────────────────────────────────────────────────────────

def test_mark_started_sets_in_progress(mgr):
    mgr.create(CONFIG)
    mgr.mark_started("EKF_Freq", "G1_E1")
    assert mgr.in_progress is not None
    assert mgr.in_progress["pair"] == "EKF_Freq::G1_E1"


def test_mark_completed_clears_in_progress(mgr):
    mgr.create(CONFIG)
    mgr.mark_started("EKF_Freq", "G1_E1")
    mgr.mark_completed("EKF_Freq", "G1_E1", result_file="out.json", n_mc=10)
    assert mgr.in_progress is None


def test_is_completed_true_after_mark_completed(mgr):
    mgr.create(CONFIG)
    assert not mgr.is_completed("EKF_Freq", "G1_E1")
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    assert mgr.is_completed("EKF_Freq", "G1_E1")


def test_mark_failed_stores_error(mgr):
    mgr.create(CONFIG)
    mgr.mark_failed("RDFT", "IBR_Nightmare", "division by zero")
    assert mgr.is_failed("RDFT", "IBR_Nightmare")
    failed = dict(mgr.list_failed_pairs())
    assert "RDFT::IBR_Nightmare" in failed
    assert "division by zero" in failed["RDFT::IBR_Nightmare"]


def test_mark_completed_removes_from_failed(mgr):
    mgr.create(CONFIG)
    mgr.mark_failed("RDFT", "IBR_Nightmare", "err")
    assert mgr.is_failed("RDFT", "IBR_Nightmare")
    mgr.mark_completed("RDFT", "IBR_Nightmare", n_mc=5)
    assert not mgr.is_failed("RDFT", "IBR_Nightmare")
    assert mgr.is_completed("RDFT", "IBR_Nightmare")


# ── should_run ────────────────────────────────────────────────────────────────

def test_should_run_restart_always_true(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    assert mgr.should_run("EKF_Freq", "G1_E1", "restart") is True


def test_should_run_resume_skips_completed(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    assert mgr.should_run("EKF_Freq", "G1_E1", "resume") is False


def test_should_run_resume_runs_incomplete(mgr):
    mgr.create(CONFIG)
    assert mgr.should_run("RDFT", "G1_E1", "resume") is True


def test_should_run_partial_skips_completed(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    assert mgr.should_run("EKF_Freq", "G1_E1", "partial") is False


def test_should_run_partial_runs_failed(mgr):
    mgr.create(CONFIG)
    mgr.mark_failed("RDFT", "G1_E1", "err")
    assert mgr.should_run("RDFT", "G1_E1", "partial") is True


# ── Atomic write ──────────────────────────────────────────────────────────────

def test_atomic_write_no_partial_files(mgr, tmp_path):
    """After mark_completed the .tmp file must not exist."""
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


def test_file_is_valid_json_after_each_operation(mgr, tmp_path):
    mgr.create(CONFIG)
    mgr.mark_started("A", "B")
    mgr.mark_completed("A", "B", n_mc=3)
    mgr.mark_started("C", "D")
    mgr.mark_failed("C", "D", "oops")
    cp_file = tmp_path / ".ofb_checkpoint.json"
    with open(cp_file) as fh:
        data = json.load(fh)
    assert "completed" in data
    assert "failed" in data


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_removes_file(mgr, tmp_path):
    mgr.create(CONFIG)
    assert (tmp_path / ".ofb_checkpoint.json").exists()
    mgr.delete()
    assert not (tmp_path / ".ofb_checkpoint.json").exists()


def test_delete_resets_state(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    mgr.delete()
    assert mgr.load() is None
    assert mgr.count_completed() == 0


# ── Counts ────────────────────────────────────────────────────────────────────

def test_count_completed(mgr):
    mgr.create(CONFIG)
    assert mgr.count_completed() == 0
    mgr.mark_completed("A", "B", n_mc=5)
    mgr.mark_completed("C", "D", n_mc=5)
    assert mgr.count_completed() == 2


def test_count_failed(mgr):
    mgr.create(CONFIG)
    mgr.mark_failed("A", "B", "err")
    assert mgr.count_failed() == 1


def test_list_completed_pairs(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    mgr.mark_completed("RDFT", "G1_E1", n_mc=10)
    pairs = mgr.list_completed_pairs()
    assert "EKF_Freq::G1_E1" in pairs
    assert "RDFT::G1_E1" in pairs


# ── Startup dialog (non-interactive path) ─────────────────────────────────────

def test_startup_dialog_no_checkpoint_creates_fresh(mgr):
    mode = mgr.startup_dialog(CONFIG, n_total=4)
    assert mode == "restart"
    assert mgr.load() is not None


def test_startup_dialog_auto_resume(mgr):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    mode = mgr.startup_dialog(CONFIG, n_total=4, auto_resume=True)
    assert mode == "resume"


def test_startup_dialog_auto_restart(mgr, tmp_path):
    mgr.create(CONFIG)
    mgr.mark_completed("EKF_Freq", "G1_E1", n_mc=10)
    mode = mgr.startup_dialog(CONFIG, n_total=4, auto_restart=True)
    assert mode == "restart"
    # Old completed entries are gone (fresh checkpoint)
    assert mgr.count_completed() == 0


# ── Pair key helper ───────────────────────────────────────────────────────────

def test_pair_key_format():
    assert _pair_key("EKF_Freq", "G1_E1") == "EKF_Freq::G1_E1"
