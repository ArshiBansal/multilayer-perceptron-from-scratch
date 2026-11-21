"""Regression tests for experiment-comparison visualizations."""

from __future__ import annotations

from src.experiment_visualization import make_experiment_tradeoff_3d


def test_runtime_tradeoff_includes_rows_without_parameter_counts() -> None:
    rows = [
        {
            "name": "runtime-only",
            "group": "runtime",
            "val_accuracy": 0.95,
            "val_loss": 0.2,
            "seconds": 12.5,
        },
        {
            "name": "missing-runtime",
            "group": "runtime",
            "val_accuracy": 0.96,
            "val_loss": 0.1,
        },
    ]

    figure = make_experiment_tradeoff_3d(rows, depth_metric="seconds")

    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [12.5]
    assert list(figure.data[0].text) == ["runtime-only"]


def test_parameter_tradeoff_excludes_rows_without_parameter_counts() -> None:
    rows = [
        {
            "name": "parameterized",
            "group": "models",
            "val_accuracy": 0.95,
            "val_loss": 0.2,
            "parameters": 1000,
        },
        {
            "name": "runtime-only",
            "group": "models",
            "val_accuracy": 0.96,
            "val_loss": 0.1,
            "seconds": 12.5,
        },
    ]

    figure = make_experiment_tradeoff_3d(rows, depth_metric="parameters")

    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [1000]
    assert list(figure.data[0].text) == ["parameterized"]
