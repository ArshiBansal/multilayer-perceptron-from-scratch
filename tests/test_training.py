"""Tests for deterministic data splitting and model training helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import split_train_validation
from src.model import NumpyMLP
from src.train import TrainingConfig, evaluate_model, train_model


def _synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(3, dtype=np.int64), 6)
    images = np.random.default_rng(7).integers(
        0,
        256,
        size=(len(labels), 28, 28),
        dtype=np.uint8,
    )
    return images, labels


def _small_model(seed: int = 5) -> NumpyMLP:
    return NumpyMLP(
        input_dim=784,
        hidden_dims=(5,),
        output_dim=3,
        seed=seed,
    )


def test_split_train_validation_is_deterministic_and_stratified() -> None:
    images, labels = _synthetic_dataset()

    first_train, first_validation = split_train_validation(
        images, labels, validation_size=6, seed=17
    )
    second_train, second_validation = split_train_validation(
        images, labels, validation_size=6, seed=17
    )

    np.testing.assert_array_equal(first_train[0], second_train[0])
    np.testing.assert_array_equal(first_train[1], second_train[1])
    np.testing.assert_array_equal(first_validation[0], second_validation[0])
    np.testing.assert_array_equal(first_validation[1], second_validation[1])
    assert len(first_train[1]) == 12
    assert len(first_validation[1]) == 6
    np.testing.assert_array_equal(np.bincount(first_validation[1]), [2, 2, 2])


def test_split_train_validation_rejects_mismatched_labels() -> None:
    images, labels = _synthetic_dataset()

    with pytest.raises(ValueError, match="same length"):
        split_train_validation(images, labels[:-1], validation_size=6)


def test_evaluate_model_is_deterministic() -> None:
    images, labels = _synthetic_dataset()
    model = _small_model()

    first = evaluate_model(model, images, labels, batch_size=4)
    second = evaluate_model(model, images, labels, batch_size=4)

    assert first == second
    assert np.isfinite(first.loss)
    assert 0.0 <= first.accuracy <= 1.0


def test_evaluate_model_rejects_mismatched_labels() -> None:
    images, labels = _synthetic_dataset()

    with pytest.raises(ValueError, match="evaluation labels must have shape"):
        evaluate_model(_small_model(), images, labels[:-1])


def test_train_model_synthetic_run_is_reproducible() -> None:
    images, labels = _synthetic_dataset()
    train_images, train_labels = images[:12], labels[:12]
    validation_images, validation_labels = images[12:], labels[12:]
    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=0.05,
        seed=11,
    )
    first_model = _small_model()
    second_model = _small_model()

    first_history = train_model(
        first_model,
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        config=config,
        verbose=False,
    )
    train_model(
        second_model,
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        config=config,
        verbose=False,
    )

    assert len(first_history.epochs) == 2
    assert [record.epoch for record in first_history.epochs] == [1, 2]
    for name in first_model.params:
        np.testing.assert_array_equal(
            first_model.params[name],
            second_model.params[name],
        )
    for record in first_history.epochs:
        assert np.isfinite(
            [
                record.train_loss,
                record.train_accuracy,
                record.validation_loss,
                record.validation_accuracy,
                record.elapsed_seconds,
            ]
        ).all()
        assert 0.0 <= record.train_accuracy <= 1.0
        assert 0.0 <= record.validation_accuracy <= 1.0


def test_train_model_rejects_mismatched_training_labels() -> None:
    images, labels = _synthetic_dataset()

    with pytest.raises(ValueError, match="training labels must have shape"):
        train_model(
            _small_model(),
            images[:12],
            labels[:11],
            images[12:],
            labels[12:],
            config=TrainingConfig(epochs=1, batch_size=4),
            verbose=False,
        )