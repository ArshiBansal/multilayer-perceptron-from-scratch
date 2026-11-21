"""Tests for model artifact companion-path mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts import (
    loss_landscape_path,
    validate_loss_landscape_model,
)


def test_default_model_uses_legacy_landscape_path() -> None:
    assert loss_landscape_path(
        Path("models") / "mnist_mlp.npz",
        Path("reports") / "figures",
    ) == Path("reports") / "figures" / "loss_surface.npz"


def test_custom_model_uses_model_specific_landscape_path() -> None:
    assert loss_landscape_path(
        Path("models") / "mnist_mlp_256_128.npz",
        Path("reports") / "figures",
    ) == Path("reports") / "figures" / "mnist_mlp_256_128_loss_surface.npz"


def test_loss_landscape_path_requires_numpy_model_artifact() -> None:
    with pytest.raises(ValueError, match=r"\.npz extension"):
        loss_landscape_path("models/mnist_mlp_256_128.pt", "reports/figures")


def test_custom_landscape_must_record_matching_model() -> None:
    validate_loss_landscape_model(
        "models/mnist_mlp_256_128.npz",
        "mnist_mlp_256_128.npz",
    )
    with pytest.raises(ValueError, match="not 'mnist_mlp_256_128.npz'"):
        validate_loss_landscape_model(
            "models/mnist_mlp_256_128.npz",
            "mnist_mlp.npz",
        )


def test_legacy_landscape_without_model_identity_is_default_only() -> None:
    validate_loss_landscape_model("models/mnist_mlp.npz", None)
    with pytest.raises(ValueError, match="does not record its source model"):
        validate_loss_landscape_model("models/mnist_mlp_256_128.npz", None)