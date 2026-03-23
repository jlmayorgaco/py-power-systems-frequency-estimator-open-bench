"""Smoke test: ofb doctor command."""

from openfreqbench.cli.app import app
from typer.testing import CliRunner


def test_doctor_exits_ok():
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
