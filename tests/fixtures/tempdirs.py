"""Temporary directory fixtures for artifact I/O tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"
