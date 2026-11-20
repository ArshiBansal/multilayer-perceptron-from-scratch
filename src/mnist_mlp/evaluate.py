"""Evaluation metrics for the NumPy MNIST MLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import IMAGE_SHAPE, INPUT_DIM, preprocess_images
from .losses import cross_entropy_from_logits
from .model import NumpyMLP


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate and per-example results for a classification evaluation."""

    loss: float
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    f1_score: NDArray[np.float64]
    support: NDArray[np.int64]
    labels: NDArray[np.intp]
    predictions: NDArray[np.intp]
    probabilities: NDArray[np.float32] | NDArray[np.float64]

    @property
    def misclassified_indices(self) -> NDArray[np.intp]:
        """Indices of examples classified incorrectly."""
        return np.flatnonzero(self.labels != self.predictions)

    def summary(self) -> dict[str, float]:
        """Return overall and macro-averaged metrics."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_precision": float(np.mean(self.precision)),
            "macro_recall": float(np.mean(self.recall)),
            "macro_f1": float(np.mean(self.f1_score)),
        }


def evaluate(
    model: NumpyMLP,
    images: ArrayLike,
    labels: ArrayLike,
    *,
    batch_size: int = 512,
) -> EvaluationReport:
    """Evaluate a classifier on raw grayscale images.

    Images must contain pixel values in [0, 255], shaped (N, 28, 28) or
    (N, 784). Labels must be integer class IDs. The function does not load data;
    pass the held-out test split only after training and model selection.

    Args:
        model: Trained NumPy MLP.
        images: Raw grayscale images.
        labels: Integer class labels.
        batch_size: Number of examples processed per forward pass.

    Returns:
        An ``EvaluationReport`` with aggregate metrics and per-example outputs.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if model.input_dim != INPUT_DIM:
        raise ValueError(f"MNIST evaluation requires input_dim={INPUT_DIM}.")

    image_array = np.asarray(images)
    label_array = np.asarray(labels)

    valid_image_shape = (
        image_array.ndim == 3 and image_array.shape[1:] == IMAGE_SHAPE
    ) or (
        image_array.ndim == 2 and image_array.shape[1] == model.input_dim
    )
    if not valid_image_shape or len(image_array) == 0:
        raise ValueError(
            "images must have shape (N, 28, 28) or (N, 784), with N > 0; "
            f"got {image_array.shape}."
        )

    if label_array.ndim != 1 or len(label_array) != len(image_array):
        raise ValueError(
            f"labels must have shape ({len(image_array)},); "
            f"got {label_array.shape}."
        )
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("labels must have an integer dtype.")
    if np.any(label_array < 0) or np.any(label_array >= model.output_dim):
        raise ValueError(
            f"labels must be in [0, {model.output_dim - 1}]."
        )

    integer_labels = label_array.astype(np.intp, copy=False)
    probabilities = np.empty(
        (len(integer_labels), model.output_dim),
        dtype=model.dtype,
    )
    confusion = np.zeros(
        (model.output_dim, model.output_dim),
        dtype=np.int64,
    )
    total_loss = 0.0
    total_correct = 0

    for start in range(0, len(integer_labels), batch_size):
        stop = min(start + batch_size, len(integer_labels))
        X_batch = preprocess_images(
            image_array[start:stop],
            flatten=True,
            normalize=True,
        )
        y_batch = integer_labels[start:stop]

        logits, _ = model.forward(X_batch)
        batch_probabilities = model.predict_proba(X_batch)
        batch_predictions = np.argmax(batch_probabilities, axis=1)

        probabilities[start:stop] = batch_probabilities
        batch_count = stop - start
        total_loss += float(cross_entropy_from_logits(logits, y_batch)) * batch_count
        total_correct += int(np.sum(batch_predictions == y_batch))
        np.add.at(confusion, (y_batch, batch_predictions), 1)

    true_positives = np.diag(confusion).astype(np.float64)
    predicted_counts = confusion.sum(axis=0)
    support = confusion.sum(axis=1)

    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros_like(true_positives),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros_like(true_positives),
        where=support != 0,
    )
    f1_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    predictions = np.argmax(probabilities, axis=1).astype(np.intp, copy=False)

    return EvaluationReport(
        loss=total_loss / len(integer_labels),
        accuracy=total_correct / len(integer_labels),
        confusion_matrix=confusion,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        support=support,
        labels=integer_labels.copy(),
        predictions=predictions,
        probabilities=probabilities,
    )