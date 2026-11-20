"""Activation functions and their derivatives for the NumPy MLP."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]

__all__ = [
    "relu",
    "relu_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "tanh",
    "tanh_derivative",
    "softmax",
    "softmax_jacobian",
]


def _as_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Validate real numeric input and return an array with a floating dtype."""
    array = np.asarray(values)

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must not contain complex values.")

    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64)
    elif array.dtype == np.float16:
        array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")

    return array


def relu(values: ArrayLike) -> FloatArray:
    """Apply ReLU element-wise: max(0, x)."""
    array = _as_float_array(values, name="values")
    return np.maximum(array, 0)


def relu_derivative(pre_activations: ArrayLike) -> FloatArray:
    """Return the ReLU derivative; the derivative at zero is defined as 0."""
    array = _as_float_array(pre_activations, name="pre_activations")
    return (array > 0).astype(array.dtype)


def sigmoid(values: ArrayLike) -> FloatArray:
    """Apply the logistic sigmoid with overflow-safe evaluation."""
    array = _as_float_array(values, name="values")
    result = np.empty_like(array)

    nonnegative = array >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))

    exp_values = np.exp(array[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)

    return result


def sigmoid_derivative(pre_activations: ArrayLike) -> FloatArray:
    """Return the sigmoid derivative evaluated from pre-activations."""
    activated = sigmoid(pre_activations)
    return activated * (1.0 - activated)


def tanh(values: ArrayLike) -> FloatArray:
    """Apply hyperbolic tangent element-wise."""
    array = _as_float_array(values, name="values")
    return np.tanh(array)


def tanh_derivative(pre_activations: ArrayLike) -> FloatArray:
    """Return the tanh derivative evaluated from pre-activations."""
    activated = tanh(pre_activations)
    return 1.0 - activated**2


def softmax(
    logits: ArrayLike,
    *,
    axis: int = -1,
) -> FloatArray:
    """Convert logits to probabilities along the selected axis.

    Subtracting the maximum logit along the axis improves numerical stability.
    """
    array = _as_float_array(logits, name="logits")

    if array.ndim == 0:
        raise ValueError("softmax requires an array with at least one dimension.")
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer.")
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for a {array.ndim}D array.")

    normalized_axis = int(axis) % array.ndim
    if array.shape[normalized_axis] == 0:
        raise ValueError("softmax cannot operate on an empty class dimension.")

    shifted = array - np.max(array, axis=normalized_axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=normalized_axis, keepdims=True)


def softmax_jacobian(probabilities: ArrayLike) -> FloatArray:
    """Return the softmax Jacobian for each row of class probabilities.

    For probabilities shaped ``(..., classes)``, the result is shaped
    ``(..., classes, classes)``. For softmax followed by cross-entropy, use
    the simplified gradient ``probabilities - one_hot_labels`` instead of
    constructing this Jacobian.
    """
    array = _as_float_array(probabilities, name="probabilities")

    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("probabilities must have a non-empty class dimension.")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("probabilities must be in the range [0, 1].")
    if not np.allclose(array.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("probabilities must sum to 1 along the last axis.")

    diagonal = np.eye(array.shape[-1], dtype=array.dtype) * array[..., :, None]
    outer_product = array[..., :, None] * array[..., None, :]
    return diagonal - outer_product