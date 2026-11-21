"""Regression tests for commands run directly from a checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.train import build_parser

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


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--learning-rate", "nan"),
        ("--learning-rate", "inf"),
        ("--learning-rate", "-inf"),
        ("--l2-strength", "nan"),
        ("--l2-strength", "inf"),
        ("--l2-strength", "-inf"),
        ("--momentum", "nan"),
        ("--momentum", "inf"),
        ("--momentum", "-inf"),
    ],
)
def test_float_options_reject_non_finite_values(
    option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([f"{option}={value}"])

    assert exc_info.value.code == 2
    assert "must be finite" in capsys.readouterr().err


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("0.999999", 0.999999)])
def test_momentum_accepts_values_at_valid_range_boundaries(
    value: str, expected: float
) -> None:
    args = build_parser().parse_args([f"--momentum={value}"])

    assert args.momentum == expected


def test_momentum_rejects_exclusive_upper_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--momentum=1"])

    assert exc_info.value.code == 2
    assert "must be in the range [0, 1)" in capsys.readouterr().err