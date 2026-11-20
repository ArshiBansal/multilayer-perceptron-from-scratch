"""Pure-NumPy multilayer perceptron for classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ActivationName = Literal["relu", "sigmoid"]
InitializationName = Literal["he", "xavier"]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ForwardCache:
    """Intermediate arrays required to calculate model gradients."""

    activations: tuple[NDArray[np.floating], ...]
    pre_activations: tuple[NDArray[np.floating], ...]


def softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute stable row-wise softmax probabilities."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be a non-empty 2D array.")
    if not np.issubdtype(logits.dtype, np.number):
        raise TypeError("logits must be numeric.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain non-finite values.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def cross_entropy_from_logits(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
) -> float:
    """Return mean multiclass cross-entropy for integer class labels."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be a non-empty 2D array.")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must have shape (number_of_examples,).")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must have an integer dtype.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain non-finite values.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain a class index outside the logits range.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    row_indices = np.arange(len(labels))
    losses = log_normalizer - shifted[row_indices, labels]
    return float(np.mean(losses))


class NumpyMLP:
    """Fully connected classifier with a NumPy forward and backward pass.

    Args:
        input_dim: Number of input features.
        hidden_dims: Width of each hidden layer.
        output_dim: Number of output classes.
        activation: Activation used in hidden layers.
        initialization: Weight initialization strategy.
        seed: Seed for reproducible parameter initialization.
        dtype: Floating-point dtype used for parameters and calculations.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        output_dim: int = 10,
        *,
        activation: ActivationName = "relu",
        initialization: InitializationName = "he",
        seed: int = 42,
        dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'.")
        if initialization not in ("he", "xavier"):
            raise ValueError("initialization must be 'he' or 'xavier'.")

        parameter_dtype = np.dtype(dtype)
        if parameter_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be np.float32 or np.float64.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.output_dim = int(output_dim)
        self.activation = activation
        self.initialization = initialization
        self.dtype = parameter_dtype

        layer_dims = (self.input_dim, *self.hidden_dims, self.output_dim)
        rng = np.random.default_rng(seed)
        self.params: dict[str, NDArray[np.floating]] = {}

        for layer_index, (fan_in, fan_out) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:], strict=False),
            start=1,
        ):
            is_hidden_layer = layer_index < len(layer_dims) - 1
            use_he = is_hidden_layer and self.initialization == "he"
            scale = np.sqrt((2.0 if use_he else 1.0) / fan_in)

            self.params[f"W{layer_index}"] = (
                rng.standard_normal((fan_in, fan_out)).astype(self.dtype) * scale
            )
            self.params[f"b{layer_index}"] = np.zeros(fan_out, dtype=self.dtype)

    @property
    def layer_count(self) -> int:
        """Number of affine layers, including the output layer."""
        return len(self.hidden_dims) + 1

    def _validate_inputs(self, X: NDArray[np.generic]) -> NDArray[np.floating]:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (N, {self.input_dim}) with N > 0; "
                f"got {X.shape}."
            )
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric values.")
        if np.issubdtype(X.dtype, np.complexfloating):
            raise TypeError("X must contain real numeric values.")

        X_array = np.asarray(X, dtype=self.dtype)
        if not np.isfinite(X_array).all():
            raise ValueError("X contains non-finite values.")
        return X_array

    def _validate_labels(
        self,
        labels: NDArray[np.generic],
        *,
        expected_size: int,
    ) -> NDArray[np.integer]:
        if labels.ndim != 1 or len(labels) != expected_size:
            raise ValueError(f"labels must have shape ({expected_size},).")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("labels must have an integer dtype.")
        if np.any(labels < 0) or np.any(labels >= self.output_dim):
            raise ValueError(
                f"labels must be in [0, {self.output_dim - 1}]."
            )
        return labels.astype(np.intp, copy=False)

    def _activate(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.activation == "relu":
            return np.maximum(values, 0)
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))

    def _activation_gradient(
        self,
        pre_activations: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        if self.activation == "relu":
            return (pre_activations > 0).astype(self.dtype)

        activated = self._activate(pre_activations)
        return activated * (1.0 - activated)

    def forward(
        self,
        X: NDArray[np.generic],
    ) -> tuple[NDArray[np.floating], ForwardCache]:
        """Return output logits and a cache for backpropagation."""
        inputs = self._validate_inputs(X)
        activations: list[NDArray[np.floating]] = [inputs]
        pre_activations: list[NDArray[np.floating]] = []

        for layer_index in range(1, self.layer_count + 1):
            values = (
                activations[-1] @ self.params[f"W{layer_index}"]
                + self.params[f"b{layer_index}"]
            )
            pre_activations.append(values)

            if layer_index < self.layer_count:
                activations.append(self._activate(values))

        cache = ForwardCache(
            activations=tuple(activations),
            pre_activations=tuple(pre_activations),
        )
        return pre_activations[-1], cache

    def loss(
        self,
        X: NDArray[np.generic],
        labels: NDArray[np.generic],
        *,
        l2_strength: float = 0.0,
    ) -> float:
        """Return mean cross-entropy with optional L2 weight regularization."""
        self._validate_l2_strength(l2_strength)
        validated_labels = self._validate_labels(
            labels,
            expected_size=len(X),
        )
        logits, _ = self.forward(X)
        data_loss = cross_entropy_from_logits(logits, validated_labels)

        if l2_strength == 0:
            return data_loss

        squared_weights = sum(
            float(np.sum(value * value))
            for name, value in self.params.items()
            if name.startswith("W")
        )
        return data_loss + 0.5 * l2_strength * squared_weights

    def backward(
        self,
        labels: NDArray[np.generic],
        cache: ForwardCache,
        *,
        l2_strength: float = 0.0,
    ) -> dict[str, NDArray[np.floating]]:
        """Compute gradients of mean cross-entropy for all parameters.

        The returned gradients include the derivative of the optional L2
        penalty. Apply updates in the training loop, for example:
        ``parameter -= learning_rate * gradients[name]``.
        """
        self._validate_l2_strength(l2_strength)

        if len(cache.activations) != self.layer_count:
            raise ValueError("cache does not match this model's architecture.")
        if len(cache.pre_activations) != self.layer_count:
            raise ValueError("cache does not match this model's architecture.")

        batch_size = len(cache.activations[0])
        validated_labels = self._validate_labels(
            labels,
            expected_size=batch_size,
        )

        logits = cache.pre_activations[-1]
        d_values = softmax(logits)
        d_values[np.arange(batch_size), validated_labels] -= 1.0
        d_values /= batch_size

        gradients: dict[str, NDArray[np.floating]] = {}

        for layer_index in range(self.layer_count, 0, -1):
            weights = self.params[f"W{layer_index}"]
            layer_input = cache.activations[layer_index - 1]

            gradients[f"W{layer_index}"] = (
                layer_input.T @ d_values + l2_strength * weights
            )
            gradients[f"b{layer_index}"] = np.sum(d_values, axis=0)

            if layer_index > 1:
                d_activations = d_values @ weights.T
                previous_z = cache.pre_activations[layer_index - 2]
                d_values = d_activations * self._activation_gradient(previous_z)

        return gradients

    def predict_proba(
        self,
        X: NDArray[np.generic],
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.floating]:
        """Return class probabilities, optionally computed in batches."""
        X_array = self._validate_inputs(X)

        if batch_size is None:
            logits, _ = self.forward(X_array)
            return softmax(logits)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        probabilities = np.empty(
            (len(X_array), self.output_dim),
            dtype=self.dtype,
        )
        for start in range(0, len(X_array), batch_size):
            stop = min(start + batch_size, len(X_array))
            logits, _ = self.forward(X_array[start:stop])
            probabilities[start:stop] = softmax(logits)

        return probabilities

    def predict(
        self,
        X: NDArray[np.generic],
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.intp]:
        """Return the most likely class for each example."""
        return np.argmax(
            self.predict_proba(X, batch_size=batch_size),
            axis=1,
        )

    @staticmethod
    def _validate_l2_strength(l2_strength: float) -> None:
        if not np.isfinite(l2_strength) or l2_strength < 0:
            raise ValueError("l2_strength must be finite and non-negative.")

    def save(
        self,
        path: str | Path,
        *,
        pixel_scale: float = 255.0,
        metadata: Mapping[str, int | float | str | bool] | None = None,
    ) -> Path:
        """Save parameters and architecture to a compressed NumPy archive."""
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("Model path must have an .npz extension.")
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError("pixel_scale must be finite and positive.")

        payload: dict[str, NDArray[np.generic]] = {
            "format_version": np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            "input_dim": np.asarray(self.input_dim, dtype=np.int64),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int64),
            "output_dim": np.asarray(self.output_dim, dtype=np.int64),
            "activation": np.asarray(self.activation),
            "initialization": np.asarray(self.initialization),
            "pixel_scale": np.asarray(pixel_scale, dtype=np.float32),
        }
        payload.update(
            {name: value for name, value in self.params.items()}
        )

        if metadata:
            for name, value in metadata.items():
                if not name.isidentifier():
                    raise ValueError(
                        f"Metadata key must be a valid identifier; got {name!r}."
                    )
                if isinstance(value, (list, tuple, dict, set)):
                    raise TypeError("Metadata values must be scalar.")
                payload[f"metadata_{name}"] = np.asarray(value)

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> NumpyMLP:
        """Load a model from a safe, non-pickle NumPy archive.

        Supports both this class's format and the earlier single-hidden-layer
        artifact containing ``hidden_dim`` instead of ``hidden_dims``.
        """
        source = Path(path)
        with np.load(source, allow_pickle=False) as artifact:
            required = {"input_dim", "output_dim", "W1", "b1"}
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError(
                    f"Model artifact is missing keys: {sorted(missing)}."
                )

            input_dim = int(artifact["input_dim"].item())
            output_dim = int(artifact["output_dim"].item())

            if "hidden_dims" in artifact.files:
                hidden_dims = tuple(
                    int(width) for width in artifact["hidden_dims"].tolist()
                )
            elif "hidden_dim" in artifact.files:
                hidden_dims = (int(artifact["hidden_dim"].item()),)
            else:
                raise ValueError(
                    "Model artifact must contain 'hidden_dims' or 'hidden_dim'."
                )

            activation = (
                str(artifact["activation"].item())
                if "activation" in artifact.files
                else "relu"
            )
            initialization = (
                str(artifact["initialization"].item())
                if "initialization" in artifact.files
                else "he"
            )

            if activation not in ("relu", "sigmoid"):
                raise ValueError(f"Unsupported saved activation: {activation!r}.")
            if initialization not in ("he", "xavier"):
                raise ValueError(
                    f"Unsupported saved initialization: {initialization!r}."
                )

            first_weight = artifact["W1"]
            if first_weight.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError("Saved model weights must be float32 or float64.")

            model = cls(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,  # type: ignore[arg-type]
                initialization=initialization,  # type: ignore[arg-type]
                dtype=first_weight.dtype.type,
            )

            for layer_index in range(1, model.layer_count + 1):
                for parameter_name in (f"W{layer_index}", f"b{layer_index}"):
                    if parameter_name not in artifact.files:
                        raise ValueError(
                            f"Model artifact is missing {parameter_name!r}."
                        )

                    value = artifact[parameter_name]
                    expected_shape = model.params[parameter_name].shape
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"{parameter_name} has shape {value.shape}; "
                            f"expected {expected_shape}."
                        )
                    if not np.isfinite(value).all():
                        raise ValueError(
                            f"Saved parameter {parameter_name} contains "
                            "non-finite values."
                        )

                    model.params[parameter_name] = value.astype(
                        model.dtype,
                        copy=True,
                    )

        return model