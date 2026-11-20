"""Finite-difference checks for model and loss gradients."""

from __future__ import annotations

import numpy as np
import pytest

from src.losses import (
    cross_entropy_from_logits,
    cross_entropy_gradient_from_logits,
)
from src.model import NumpyMLP


def _finite_difference_gradient(
    function,
    values: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Estimate a scalar function's gradient with centered differences."""
    gradient = np.zeros_like(values, dtype=np.float64)

    for index in np.ndindex(values.shape):
        original = values[index]

        values[index] = original + epsilon
        loss_plus = function()

        values[index] = original - epsilon
        loss_minus = function()

        values[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient


@pytest.mark.parametrize(
    ("hidden_dims", "activation"),
    [
        ((4,), "relu"),
        ((4, 3), "relu"),
        ((4, 3), "sigmoid"),
    ],
)
def test_model_backward_matches_finite_differences(
    hidden_dims: tuple[int, ...],
    activation: str,
) -> None:
    rng = np.random.default_rng(0)
    model = NumpyMLP(
        input_dim=3,
        hidden_dims=hidden_dims,
        output_dim=3,
        activation=activation,
        initialization="xavier",
        seed=17,
        dtype=np.float64,
    )

    X = rng.uniform(-0.7, 0.9, size=(4, 3)).astype(np.float64)
    labels = np.array([0, 2, 1, 2], dtype=np.int64)
    l2_strength = 0.03

    logits, cache = model.forward(X)

    if activation == "relu":
        # Avoid checking finite differences at ReLU's non-differentiable point.
        assert all(
            np.all(np.abs(values) > 1e-5)
            for values in cache.pre_activations[:-1]
        )

    analytical = model.backward(
        labels,
        cache,
        l2_strength=l2_strength,
    )

    for name, parameter in model.params.items():
        numerical = _finite_difference_gradient(
            lambda: model.loss(X, labels, l2_strength=l2_strength),
            parameter,
        )
        np.testing.assert_allclose(
            analytical[name],
            numerical,
            rtol=2e-5,
            atol=2e-7,
            err_msg=f"Gradient check failed for {name}.",
        )

    assert np.isfinite(logits).all()


@pytest.mark.parametrize(
    "targets",
    [
        np.array([2, 0, 1], dtype=np.int64),
        np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.7, 0.1],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_cross_entropy_gradient_matches_finite_differences(
    targets: np.ndarray,
) -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(3, 3)).astype(np.float64)

    analytical = cross_entropy_gradient_from_logits(logits, targets)

    numerical = _finite_difference_gradient(
        lambda: cross_entropy_from_logits(logits, targets),
        logits,
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-6,
        atol=1e-8,
    )


def test_cross_entropy_gradient_reductions_are_consistent() -> None:
    logits = np.array(
        [[1.2, -0.3, 0.7], [-0.5, 0.8, 0.1]],
        dtype=np.float64,
    )
    labels = np.array([0, 2], dtype=np.int64)

    mean_gradient = cross_entropy_gradient_from_logits(
        logits,
        labels,
        reduction="mean",
    )
    sum_gradient = cross_entropy_gradient_from_logits(
        logits,
        labels,
        reduction="sum",
    )

    np.testing.assert_allclose(
        mean_gradient * len(labels),
        sum_gradient,
        rtol=1e-12,
        atol=1e-12,
    )

    per_example_gradient = cross_entropy_gradient_from_logits(
        logits,
        labels,
        reduction="none",
    )
    assert per_example_gradient.shape == logits.shape
    np.testing.assert_allclose(
        per_example_gradient.sum(axis=0),
        sum_gradient.sum(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_model_backward_rejects_mismatched_label_count() -> None:
    model = NumpyMLP(
        input_dim=2,
        hidden_dims=(3,),
        output_dim=2,
        seed=4,
        dtype=np.float64,
    )
    _, cache = model.forward(np.ones((2, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="labels must have shape"):
        model.backward(np.array([1], dtype=np.int64), cache)