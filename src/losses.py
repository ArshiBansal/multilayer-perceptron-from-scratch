"""Classification loss functions and gradients for the NumPy MLP."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .activations import softmax

Reduction = Literal["none", "mean", "sum"]

__all__ = [
    "cross_entropy_from_logits",
    "cross_entropy_gradient_from_logits",
]


def _validate_reduction(reduction: str) -> None:
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _validate_logits(logits: ArrayLike) -> NDArray[np.floating]:
    array = np.asarray(logits)

    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("logits must be a non-empty 2D array.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("logits must contain real numeric values.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("logits must not contain complex values.")

    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64)
    elif array.dtype == np.float16:
        array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError("logits contain non-finite values.")

    return array


def _validate_targets(
    targets: ArrayLike,
    *,
    batch_size: int,
    num_classes: int,
    dtype: np.dtype,
) -> tuple[NDArray[np.integer] | None, NDArray[np.floating] | None]:
    """Validate labels, returning either class indices or target distributions."""
    array = np.asarray(targets)

    if array.ndim == 1:
        if len(array) != batch_size:
            raise ValueError(f"integer targets must have shape ({batch_size},).")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("One-dimensional targets must have an integer dtype.")
        if np.any(array < 0) or np.any(array >= num_classes):
            raise ValueError(f"target class indices must be in [0, {num_classes - 1}].")
        return array.astype(np.intp, copy=False), None

    if array.ndim == 2:
        if array.shape != (batch_size, num_classes):
            raise ValueError(
                "matrix targets must have shape "
                f"({batch_size}, {num_classes}); got {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Matrix targets must contain real numeric values.")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError("Matrix targets must not contain complex values.")

        distributions = array.astype(dtype, copy=False)
        if not np.isfinite(distributions).all():
            raise ValueError("Matrix targets contain non-finite values.")
        if np.any(distributions < 0):
            raise ValueError("Matrix targets must be non-negative.")
        if not np.allclose(
            distributions.sum(axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError("Each row of matrix targets must sum to 1.")

        return None, distributions

    raise ValueError(
        "targets must be integer labels shaped (N,) or distributions (N, C)."
    )


def _per_example_loss(
    logits: NDArray[np.floating],
    class_indices: NDArray[np.integer] | None,
    distributions: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Compute stable cross-entropy loss for each example."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probabilities = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    if class_indices is not None:
        return -log_probabilities[np.arange(len(logits)), class_indices]

    if distributions is None:
        raise RuntimeError("Validated targets were not provided.")

    return -np.sum(distributions * log_probabilities, axis=1)


def cross_entropy_from_logits(
    logits: ArrayLike,
    targets: ArrayLike,
    *,
    reduction: Reduction = "mean",
) -> float | NDArray[np.floating]:
    """Compute categorical cross-entropy directly from logits.

    Args:
        logits: Unnormalized model outputs shaped (N, C).
        targets: Integer class labels shaped (N,), or one-hot/soft target
            distributions shaped (N, C).
        reduction: ``"none"`` returns one loss per example; ``"mean"`` or
            ``"sum"`` returns a scalar.

    Returns:
        A scalar float for ``"mean"`` or ``"sum"``, otherwise a 1D loss array.
    """
    _validate_reduction(reduction)
    logits_array = _validate_logits(logits)
    class_indices, distributions = _validate_targets(
        targets,
        batch_size=len(logits_array),
        num_classes=logits_array.shape[1],
        dtype=logits_array.dtype,
    )
    losses = _per_example_loss(logits_array, class_indices, distributions)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses, dtype=np.float64))
    return float(np.mean(losses, dtype=np.float64))


def cross_entropy_gradient_from_logits(
    logits: ArrayLike,
    targets: ArrayLike,
    *,
    reduction: Reduction = "mean",
) -> NDArray[np.floating]:
    """Compute the gradient of softmax cross-entropy with respect to logits.

    The default ``"mean"`` reduction divides the gradient by the batch size,
    matching ``cross_entropy_from_logits(..., reduction="mean")``.
    """
    _validate_reduction(reduction)
    logits_array = _validate_logits(logits)
    class_indices, distributions = _validate_targets(
        targets,
        batch_size=len(logits_array),
        num_classes=logits_array.shape[1],
        dtype=logits_array.dtype,
    )

    gradient = softmax(logits_array)

    if class_indices is not None:
        gradient[np.arange(len(logits_array)), class_indices] -= 1.0
    elif distributions is not None:
        gradient -= distributions
    else:
        raise RuntimeError("Validated targets were not provided.")

    if reduction == "mean":
        gradient /= len(logits_array)

    return gradient