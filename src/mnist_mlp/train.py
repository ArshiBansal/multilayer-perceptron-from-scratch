"""Training and evaluation loops for the NumPy MLP."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import IMAGE_SHAPE, INPUT_DIM, NUM_CLASSES, preprocess_images
from .losses import cross_entropy_from_logits
from .model import NumpyMLP
from .optimizers import SGD


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for mini-batch SGD training."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.0
    l2_strength: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name, value in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be finite and in the range [0, 1).")
        if not np.isfinite(self.l2_strength) or self.l2_strength < 0:
            raise ValueError("l2_strength must be finite and non-negative.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Loss and accuracy over a dataset."""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class EpochMetrics:
    """Training and validation metrics recorded after one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingHistory:
    """Metrics recorded during training."""

    epochs: tuple[EpochMetrics, ...]

    def as_dict(self) -> dict[str, list[float]]:
        """Return metrics as lists, suitable for plotting or serialization."""
        return {
            "epoch": [float(item.epoch) for item in self.epochs],
            "train_loss": [item.train_loss for item in self.epochs],
            "train_accuracy": [item.train_accuracy for item in self.epochs],
            "validation_loss": [item.validation_loss for item in self.epochs],
            "validation_accuracy": [
                item.validation_accuracy for item in self.epochs
            ],
            "elapsed_seconds": [item.elapsed_seconds for item in self.epochs],
        }


def _validate_dataset(
    images: ArrayLike,
    labels: ArrayLike,
    *,
    output_dim: int,
    name: str,
) -> tuple[NDArray[np.generic], NDArray[np.integer]]:
    """Validate raw grayscale images and integer class labels."""
    image_array = np.asarray(images)
    label_array = np.asarray(labels)

    valid_image_shape = (
        image_array.ndim == 3 and image_array.shape[1:] == IMAGE_SHAPE
    ) or (image_array.ndim == 2 and image_array.shape[1] == INPUT_DIM)

    if not valid_image_shape or len(image_array) == 0:
        raise ValueError(
            f"{name} images must have shape (N, 28, 28) or (N, 784), "
            f"with N > 0; got {image_array.shape}."
        )

    if not np.issubdtype(image_array.dtype, np.number):
        raise TypeError(f"{name} images must be numeric.")
    if np.issubdtype(image_array.dtype, np.complexfloating):
        raise TypeError(f"{name} images must not be complex.")
    if not np.isfinite(image_array).all():
        raise ValueError(f"{name} images contain non-finite values.")
    if np.min(image_array) < 0 or np.max(image_array) > 255:
        raise ValueError(f"{name} pixel values must be in [0, 255].")

    if np.issubdtype(image_array.dtype, np.floating) and np.max(image_array) <= 1:
        raise ValueError(
            f"{name} images appear to be normalized already; "
            "provide raw pixel values in [0, 255]."
        )

    if label_array.ndim != 1 or len(label_array) != len(image_array):
        raise ValueError(
            f"{name} labels must have shape ({len(image_array)},); "
            f"got {label_array.shape}."
        )
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError(f"{name} labels must have an integer dtype.")
    if np.any(label_array < 0) or np.any(label_array >= output_dim):
        raise ValueError(
            f"{name} labels must be in [0, {output_dim - 1}]."
        )

    return image_array, label_array


def _make_batch(
    images: NDArray[np.generic],
    labels: NDArray[np.integer],
    indices: NDArray[np.integer],
) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    """Select and normalize a batch of raw images."""
    X_batch = preprocess_images(images[indices], flatten=True, normalize=True)
    y_batch = labels[indices].astype(np.intp, copy=False)
    return X_batch, y_batch


def _regularization_penalty(model: NumpyMLP, strength: float) -> float:
    if strength == 0:
        return 0.0

    squared_weights = sum(
        float(np.sum(weights * weights, dtype=np.float64))
        for name, weights in model.params.items()
        if name.startswith("W")
    )
    return 0.5 * strength * squared_weights


def evaluate_model(
    model: NumpyMLP,
    images: ArrayLike,
    labels: ArrayLike,
    *,
    batch_size: int = 512,
    l2_strength: float = 0.0,
) -> EvaluationMetrics:
    """Evaluate mean cross-entropy and accuracy over raw grayscale images.

    This function is intended for training or validation data. The caller
    controls when and whether the held-out test set is evaluated.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not np.isfinite(l2_strength) or l2_strength < 0:
        raise ValueError("l2_strength must be finite and non-negative.")

    image_array, label_array = _validate_dataset(
        images,
        labels,
        output_dim=model.output_dim,
        name="evaluation",
    )

    total_loss = 0.0
    total_correct = 0
    total_examples = len(label_array)

    for start in range(0, total_examples, batch_size):
        indices = np.arange(
            start,
            min(start + batch_size, total_examples),
            dtype=np.intp,
        )
        X_batch, y_batch = _make_batch(image_array, label_array, indices)
        logits, _ = model.forward(X_batch)

        count = len(indices)
        batch_loss = cross_entropy_from_logits(logits, y_batch)
        total_loss += float(batch_loss) * count
        total_correct += int(np.sum(np.argmax(logits, axis=1) == y_batch))

    mean_loss = total_loss / total_examples
    mean_loss += _regularization_penalty(model, l2_strength)

    return EvaluationMetrics(
        loss=mean_loss,
        accuracy=total_correct / total_examples,
    )


def train_model(
    model: NumpyMLP,
    train_images: ArrayLike,
    train_labels: ArrayLike,
    validation_images: ArrayLike,
    validation_labels: ArrayLike,
    *,
    config: TrainingConfig | None = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train a model using mini-batch SGD and return per-epoch metrics.

    Images must contain raw grayscale pixel values in [0, 255], shaped
    (N, 28, 28) or (N, 784). Labels must be integer class IDs.

    Training and validation metrics are computed over each complete split after
    every epoch. Keep the official test set separate and evaluate it only after
    model selection is complete.
    """
    if config is None:
        config = TrainingConfig()

    X_train, y_train = _validate_dataset(
        train_images,
        train_labels,
        output_dim=model.output_dim,
        name="training",
    )
    X_validation, y_validation = _validate_dataset(
        validation_images,
        validation_labels,
        output_dim=model.output_dim,
        name="validation",
    )

    optimizer = SGD(
        learning_rate=config.learning_rate,
        momentum=config.momentum,
    )
    rng = np.random.default_rng(config.seed)
    records: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        shuffled_indices = rng.permutation(len(y_train))

        for start in range(0, len(shuffled_indices), config.batch_size):
            batch_indices = shuffled_indices[start : start + config.batch_size]
            X_batch, y_batch = _make_batch(X_train, y_train, batch_indices)

            _, cache = model.forward(X_batch)
            gradients = model.backward(
                y_batch,
                cache,
                l2_strength=config.l2_strength,
            )
            optimizer.step(model.params, gradients)

        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
            batch_size=config.batch_size,
            l2_strength=config.l2_strength,
        )
        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            batch_size=config.batch_size,
            l2_strength=config.l2_strength,
        )

        record = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_loss=validation_metrics.loss,
            validation_accuracy=validation_metrics.accuracy,
            elapsed_seconds=time.perf_counter() - epoch_start,
        )
        records.append(record)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{config.epochs} | "
                f"train loss {record.train_loss:.4f}, "
                f"acc {record.train_accuracy:.2%} | "
                f"validation loss {record.validation_loss:.4f}, "
                f"acc {record.validation_accuracy:.2%} | "
                f"{record.elapsed_seconds:.1f}s"
            )

    return TrainingHistory(epochs=tuple(records))