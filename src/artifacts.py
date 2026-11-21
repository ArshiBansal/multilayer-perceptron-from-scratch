"""Helpers for mapping model artifacts to their companion outputs."""

from __future__ import annotations

from pathlib import Path


def loss_landscape_path(
    model_path: str | Path,
    report_dir: str | Path,
) -> Path:
    """Return the landscape data path associated with a model artifact.

    The default model keeps its historical landscape filename. Other models
    receive a filename derived from their artifact stem.
    """
    artifact_path = Path(model_path)
    if artifact_path.suffix.lower() != ".npz":
        raise ValueError("Model artifact path must have an .npz extension.")

    filename = (
        "loss_surface.npz"
        if artifact_path.name == "mnist_mlp.npz"
        else f"{artifact_path.stem}_loss_surface.npz"
    )
    return Path(report_dir) / filename


def validate_loss_landscape_model(
    model_path: str | Path,
    landscape_model: str | None,
) -> None:
    """Reject landscape data that is not identified with the selected model."""
    model_name = Path(model_path).name
    if landscape_model is not None:
        if landscape_model != model_name:
            raise ValueError(
                f"Landscape artifact was generated for {landscape_model!r}, "
                f"not {model_name!r}."
            )
    elif model_name != "mnist_mlp.npz":
        raise ValueError(
            "Landscape artifact does not record its source model; "
            "regenerate it for the selected model."
        )