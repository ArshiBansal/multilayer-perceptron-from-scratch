"""Plotly visualizations for training and interpreting the NumPy MLP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots

Normalization = Literal["true", "predicted"] | None

__all__ = [
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_first_layer_weights",
    "plot_prediction",
    "plot_loss_landscape",
    "save_figure_html",
]


def _finite_array(values: ArrayLike, *, name: str) -> NDArray[np.floating]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must not contain complex values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def plot_training_history(
    history: Mapping[str, Sequence[float]] | object,
) -> go.Figure:
    """Plot training and validation loss and accuracy by epoch.

    Accepts a mapping with keys ``train_loss``, ``validation_loss``,
    ``train_accuracy``, and ``validation_accuracy``; a ``TrainingHistory``
    instance from ``train.py`` is also accepted.
    """
    if hasattr(history, "as_dict") and callable(history.as_dict):
        data = history.as_dict()
    elif isinstance(history, Mapping):
        data = history
    else:
        raise TypeError("history must be a mapping or expose an as_dict() method.")

    required = (
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"history is missing required metrics: {missing}.")

    metrics = {key: _finite_array(data[key], name=key) for key in required}
    lengths = {len(values) for values in metrics.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("history metrics must be non-empty and have equal lengths.")

    count = next(iter(lengths))
    epochs = np.asarray(data.get("epoch", np.arange(1, count + 1)))
    if epochs.ndim != 1 or len(epochs) != count:
        raise ValueError("epoch values must match the number of metric values.")

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss", "Accuracy"),
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics["train_loss"],
            mode="lines+markers",
            name="Training loss",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics["validation_loss"],
            mode="lines+markers",
            name="Validation loss",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics["train_accuracy"],
            mode="lines+markers",
            name="Training accuracy",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics["validation_accuracy"],
            mode="lines+markers",
            name="Validation accuracy",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Epoch", row=1, col=1)
    figure.update_xaxes(title_text="Epoch", row=1, col=2)
    figure.update_yaxes(title_text="Cross-entropy loss", row=1, col=1)
    figure.update_yaxes(
        title_text="Accuracy",
        tickformat=".0%",
        range=[0, 1],
        row=1,
        col=2,
    )
    figure.update_layout(
        title="Training history",
        legend=dict(orientation="h", yanchor="bottom", y=1.08),
    )
    return figure


def plot_confusion_matrix(
    confusion_matrix: ArrayLike,
    *,
    class_names: Sequence[str] | None = None,
    normalize: Normalization = None,
) -> go.Figure:
    """Plot a square confusion matrix, optionally normalized by true or predicted class.

    Rows represent true classes and columns represent predicted classes.
    """
    matrix = _finite_array(confusion_matrix, name="confusion_matrix")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a non-empty square matrix.")
    if np.any(matrix < 0):
        raise ValueError("confusion_matrix cannot contain negative values.")
    if normalize not in (None, "true", "predicted"):
        raise ValueError("normalize must be None, 'true', or 'predicted'.")

    class_count = matrix.shape[0]
    if class_names is None:
        labels = [str(index) for index in range(class_count)]
    else:
        if len(class_names) != class_count:
            raise ValueError("class_names length must match the matrix dimensions.")
        labels = [str(label) for label in class_names]

    values = matrix.astype(np.float64, copy=True)
    if normalize == "true":
        denominators = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators != 0,
        )
    elif normalize == "predicted":
        denominators = values.sum(axis=0, keepdims=True)
        values = np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators != 0,
        )

    if normalize is None:
        text = np.rint(values).astype(np.int64).astype(str)
        colorbar_title = "Count"
        hover_format = "%{z:.0f}"
    else:
        text = np.vectorize(lambda value: f"{value:.2f}")(values)
        colorbar_title = "Proportion"
        hover_format = "%{z:.2%}"

    title_suffix = {
        None: "counts",
        "true": "normalized by true class",
        "predicted": "normalized by predicted class",
    }[normalize]

    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "True: %{y}<br>Predicted: %{x}<br>"
                f"Value: {hover_format}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Confusion matrix ({title_suffix})",
        xaxis_title="Predicted class",
        yaxis_title="True class",
        yaxis=dict(autorange="reversed"),
    )
    return figure


def plot_first_layer_weights(
    model: object,
    *,
    max_units: int = 64,
    columns: int = 8,
    image_shape: tuple[int, int] = (28, 28),
) -> go.Figure:
    """Visualize first-layer weights as a grid of grayscale image patterns."""
    if isinstance(max_units, bool) or not isinstance(max_units, int) or max_units <= 0:
        raise ValueError("max_units must be a positive integer.")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer.")
    if (
        len(image_shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
               for size in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers.")

    params = getattr(model, "params", None)
    if not isinstance(params, Mapping) or "W1" not in params:
        raise TypeError("model must expose a params mapping containing 'W1'.")

    weights = _finite_array(params["W1"], name="W1")
    if weights.ndim != 2 or weights.shape[0] != image_shape[0] * image_shape[1]:
        raise ValueError(
            f"W1 must have shape ({image_shape[0] * image_shape[1]}, hidden_units)."
        )

    unit_count = min(max_units, weights.shape[1])
    column_count = min(columns, unit_count)
    row_count = int(np.ceil(unit_count / column_count))
    limit = float(np.max(np.abs(weights[:, :unit_count])))
    if limit == 0:
        limit = 1.0

    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        subplot_titles=[f"Unit {unit}" for unit in range(unit_count)],
        horizontal_spacing=0.02,
        vertical_spacing=min(0.08, 0.5 / row_count),
    )

    for unit in range(unit_count):
        row = unit // column_count + 1
        column = unit % column_count + 1
        image = weights[:, unit].reshape(image_shape)

        figure.add_trace(
            go.Heatmap(
                z=image,
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=(unit == 0),
                colorbar=dict(title="Weight"),
                hovertemplate="Row: %{y}<br>Column: %{x}<br>Weight: %{z:.3f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.update_xaxes(showticklabels=False, showgrid=False, row=row, col=column)
        figure.update_yaxes(
            showticklabels=False,
            showgrid=False,
            autorange="reversed",
            row=row,
            col=column,
        )

    figure.update_layout(
        title=f"First-layer weights ({unit_count} hidden units)",
        height=max(300, 180 * row_count),
        width=max(500, 140 * column_count),
    )
    return figure


def plot_prediction(
    image: ArrayLike,
    probabilities: ArrayLike,
    *,
    true_label: int | None = None,
    class_names: Sequence[str] | None = None,
) -> go.Figure:
    """Display a grayscale digit beside its predicted class probabilities."""
    image_array = _finite_array(image, name="image")
    if image_array.shape == (784,):
        image_array = image_array.reshape(28, 28)
    if image_array.shape != (28, 28):
        raise ValueError("image must have shape (28, 28) or (784,).")
    if np.min(image_array) < 0 or np.max(image_array) > 255:
        raise ValueError("image pixel values must be in [0, 255].")

    probability_array = _finite_array(probabilities, name="probabilities")
    if probability_array.ndim != 1 or probability_array.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array.")
    if np.any(probability_array < 0) or np.any(probability_array > 1):
        raise ValueError("probabilities must be in [0, 1].")
    if not np.isclose(probability_array.sum(), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("probabilities must sum to 1.")

    class_count = len(probability_array)
    if class_names is None:
        labels = [str(index) for index in range(class_count)]
    else:
        if len(class_names) != class_count:
            raise ValueError("class_names length must match probabilities.")
        labels = [str(label) for label in class_names]

    predicted_index = int(np.argmax(probability_array))
    if true_label is not None and not 0 <= true_label < class_count:
        raise ValueError(f"true_label must be in [0, {class_count - 1}].")

    bar_colors = ["#4c78a8"] * class_count
    if true_label is not None:
        bar_colors[true_label] = "#2ca02c" if true_label == predicted_index else "#e45756"
    else:
        bar_colors[predicted_index] = "#2ca02c"

    title = f"Predicted: {labels[predicted_index]}"
    if true_label is not None:
        title += f" | True: {labels[true_label]}"

    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Input image", title),
    )
    figure.add_trace(
        go.Heatmap(
            z=image_array,
            colorscale="Gray",
            zmin=0,
            zmax=255,
            showscale=False,
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Pixel: %{z:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=probability_array,
            marker_color=bar_colors,
            hovertemplate="Class: %{x}<br>Probability: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
    figure.update_xaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text="Digit class", row=1, col=2)
    figure.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=2)
    figure.update_layout(title="Digit prediction", showlegend=False)
    return figure


def plot_loss_landscape(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    loss_grid: ArrayLike,
    *,
    trajectory_coordinates: ArrayLike | None = None,
    trajectory_losses: ArrayLike | None = None,
) -> go.Figure:
    """Plot a 3D loss surface and, optionally, a projected trajectory."""
    x_values = _finite_array(x_grid, name="x_grid")
    y_values = _finite_array(y_grid, name="y_grid")
    losses = _finite_array(loss_grid, name="loss_grid")

    if x_values.ndim != 2 or y_values.ndim != 2 or losses.ndim != 2:
        raise ValueError("x_grid, y_grid, and loss_grid must be 2D arrays.")
    if x_values.shape != y_values.shape or x_values.shape != losses.shape:
        raise ValueError("x_grid, y_grid, and loss_grid must have matching shapes.")

    has_coordinates = trajectory_coordinates is not None
    has_losses = trajectory_losses is not None
    if has_coordinates != has_losses:
        raise ValueError(
            "Provide both trajectory_coordinates and trajectory_losses, or neither."
        )

    figure = go.Figure(
        data=go.Surface(
            x=x_values,
            y=y_values,
            z=losses,
            colorscale="Viridis",
            opacity=0.82,
            colorbar=dict(title="Loss"),
            name="Loss surface",
        )
    )

    if has_coordinates and has_losses:
        coordinates = _finite_array(
            trajectory_coordinates,
            name="trajectory_coordinates",
        )
        trajectory_z = _finite_array(trajectory_losses, name="trajectory_losses")

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("trajectory_coordinates must have shape (N, 2).")
        if trajectory_z.ndim != 1 or len(trajectory_z) != len(coordinates):
            raise ValueError(
                "trajectory_losses must have shape matching trajectory_coordinates."
            )

        figure.add_trace(
            go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=trajectory_z,
                mode="lines+markers",
                marker=dict(size=3, color=np.arange(len(trajectory_z)), colorscale="Turbo"),
                line=dict(color="white", width=4),
                name="Projected trajectory",
            )
        )

    figure.update_layout(
        title="Loss landscape",
        scene=dict(
            xaxis_title="Direction 1",
            yaxis_title="Direction 2",
            zaxis_title="Loss",
        ),
        height=700,
    )
    return figure


def save_figure_html(
    figure: go.Figure,
    path: str | Path,
    *,
    include_plotlyjs: bool | Literal["cdn"] = "cdn",
) -> Path:
    """Write a Plotly figure as a standalone or CDN-backed HTML file."""
    destination = Path(path)
    if destination.suffix.lower() != ".html":
        raise ValueError("Figure output path must have an .html extension.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination, include_plotlyjs=include_plotlyjs)
    return destination