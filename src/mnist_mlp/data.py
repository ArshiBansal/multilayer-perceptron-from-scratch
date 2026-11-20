"""MNIST loading, validation, preprocessing, and splitting utilities."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

IMAGE_SHAPE = (28, 28)
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
PIXEL_SCALE = 255.0

ImageArray: TypeAlias = NDArray[np.uint8]
LabelArray: TypeAlias = NDArray[np.integer]
DatasetSplit: TypeAlias = tuple[ImageArray, LabelArray]
MNISTData: TypeAlias = tuple[DatasetSplit, DatasetSplit]


def load_mnist() -> MNISTData:
    """Load and validate MNIST using Keras' automatic download and cache.

    TensorFlow is imported only when this function is called, so application
    code can use this module without installing the training dependencies.

    Returns:
        ((X_train, y_train), (X_test, y_test)), with uint8 images shaped
        (N, 28, 28) and integer labels in the range 0–9.

    Raises:
        RuntimeError: If TensorFlow is not installed.
        ValueError: If the loaded data does not match the expected MNIST format.
    """
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as exc:
        raise RuntimeError(
            "Loading MNIST requires TensorFlow. Install the training "
            "dependencies with: python -m pip install -r requirements-train.txt"
        ) from exc

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    _validate_mnist_split(X_train, y_train, split_name="training")
    _validate_mnist_split(X_test, y_test, split_name="test")

    return (X_train, y_train), (X_test, y_test)


def _validate_mnist_split(
    images: NDArray[np.generic],
    labels: NDArray[np.generic],
    *,
    split_name: str,
) -> None:
    """Raise ValueError if an image/label pair is not a valid MNIST split."""
    if images.ndim != 3 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(
            f"{split_name} images must have shape (N, 28, 28); "
            f"got {images.shape}."
        )

    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError(
            f"{split_name} labels must have shape ({len(images)},); "
            f"got {labels.shape}."
        )

    if images.dtype != np.uint8:
        raise ValueError(
            f"{split_name} images must have dtype uint8; got {images.dtype}."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"{split_name} labels must have an integer dtype; got {labels.dtype}."
        )

    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError(
            f"{split_name} labels must contain digits in [0, {NUM_CLASSES - 1}]."
        )


def preprocess_images(
    images: NDArray[np.generic],
    *,
    flatten: bool = True,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Convert MNIST images to float32, optionally flattening and normalizing.

    Args:
        images: Images shaped (N, 28, 28), or (N, 784) when ``flatten=True``.
        flatten: Return vectors shaped (N, 784) instead of image-shaped arrays.
        normalize: Divide pixel values by 255 to map them to [0, 1].

    Returns:
        A new float32 array. The input array is not modified.
    """
    if images.ndim == 3:
        if images.shape[1:] != IMAGE_SHAPE:
            raise ValueError(
                f"Images must have shape (N, 28, 28); got {images.shape}."
            )
    elif images.ndim == 2 and flatten and images.shape[1] == INPUT_DIM:
        pass
    else:
        expected = "(N, 28, 28)" if not flatten else "(N, 28, 28) or (N, 784)"
        raise ValueError(f"Expected images shaped {expected}; got {images.shape}.")

    if not np.issubdtype(images.dtype, np.number):
        raise TypeError(f"Images must be numeric; got {images.dtype}.")

    if not np.isfinite(images).all():
        raise ValueError("Images contain non-finite values.")

    if images.size and (np.min(images) < 0 or np.max(images) > PIXEL_SCALE):
        raise ValueError(f"Pixel values must be in [0, {int(PIXEL_SCALE)}].")

    processed = images.astype(np.float32, copy=True)

    if normalize:
        processed /= np.float32(PIXEL_SCALE)

    if flatten:
        processed = processed.reshape(len(processed), INPUT_DIM)

    return processed


def one_hot_encode(
    labels: NDArray[np.generic],
    *,
    num_classes: int = NUM_CLASSES,
) -> NDArray[np.float32]:
    """Convert integer class labels to a float32 one-hot matrix."""
    if labels.ndim != 1:
        raise ValueError(f"Labels must be one-dimensional; got {labels.shape}.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Labels must have an integer dtype; got {labels.dtype}.")

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    if labels.size and (np.any(labels < 0) or np.any(labels >= num_classes)):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}].")

    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels.astype(np.intp, copy=False)] = 1.0
    return encoded


def split_train_validation(
    images: NDArray[np.generic],
    labels: NDArray[np.generic],
    *,
    validation_size: int | float = 5_000,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit]:
    """Create a deterministic, stratified train/validation split.

    Args:
        images: Image array whose first dimension indexes examples.
        labels: One-dimensional integer class labels.
        validation_size: Number of validation examples, or a fraction in (0, 1).
        seed: Seed for the NumPy random generator.

    Returns:
        ((X_train, y_train), (X_validation, y_validation)).
        The requested validation size is exact.
    """
    if images.ndim < 1:
        raise ValueError("images must have an example dimension.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional; got {labels.shape}.")
    if len(images) != len(labels):
        raise ValueError(
            f"images and labels must have the same length; "
            f"got {len(images)} and {len(labels)}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"labels must have an integer dtype; got {labels.dtype}.")
    if len(labels) < 2:
        raise ValueError("At least two examples are required.")

    if isinstance(validation_size, bool):
        raise TypeError("validation_size must be an integer count or a fraction.")

    if isinstance(validation_size, float):
        if not 0.0 < validation_size < 1.0:
            raise ValueError("A fractional validation_size must be in (0, 1).")
        validation_count = int(round(len(labels) * validation_size))
    elif isinstance(validation_size, (int, np.integer)):
        validation_count = int(validation_size)
    else:
        raise TypeError("validation_size must be an integer count or a float fraction.")

    if not 0 < validation_count < len(labels):
        raise ValueError(
            f"validation_size must be between 1 and {len(labels) - 1}; "
            f"got {validation_count}."
        )

    classes, class_counts = np.unique(labels, return_counts=True)
    if np.any(class_counts < 2):
        raise ValueError(
            "Stratified splitting requires at least two examples of every class."
        )
    if validation_count < len(classes):
        raise ValueError(
            "validation_size must be at least the number of classes "
            f"({len(classes)}) for a stratified split."
        )
    if len(labels) - validation_count < len(classes):
        raise ValueError(
            "The training split must contain at least one example per class."
        )

    # Allocate validation examples proportionally, while keeping at least one
    # example of each class in both splits.
    ideal_counts = validation_count * class_counts / len(labels)
    validation_counts = np.floor(ideal_counts).astype(np.intp)
    validation_counts = np.clip(validation_counts, 1, class_counts - 1)

    while int(validation_counts.sum()) < validation_count:
        eligible = validation_counts < class_counts - 1
        priorities = np.where(eligible, ideal_counts - validation_counts, -np.inf)
        validation_counts[np.argmax(priorities)] += 1

    while int(validation_counts.sum()) > validation_count:
        eligible = validation_counts > 1
        priorities = np.where(eligible, validation_counts - ideal_counts, -np.inf)
        validation_counts[np.argmax(priorities)] -= 1

    rng = np.random.default_rng(seed)
    train_parts: list[NDArray[np.intp]] = []
    validation_parts: list[NDArray[np.intp]] = []

    for class_label, class_validation_count in zip(classes, validation_counts):
        class_indices = np.flatnonzero(labels == class_label)
        rng.shuffle(class_indices)

        validation_parts.append(class_indices[:class_validation_count])
        train_parts.append(class_indices[class_validation_count:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        (images[train_indices], labels[train_indices]),
        (images[validation_indices], labels[validation_indices]),
    )