"""Regression tests for commands run directly from a checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/train.py",
        "scripts/download_data.py",
        "tests/test_model.py",
    ],
)
def test_direct_script_help_runs_from_project_root(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "usage:" in result.stdout.lower()
