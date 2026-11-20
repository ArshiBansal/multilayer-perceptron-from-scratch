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
    "plot_network_architecture_3d",
    "plot_activation_flow_animation",
    "plot_activation_flow_animation_3d",
    "plot_pixel_surface_3d",
    "plot_weight_surface_3d",
    "plot_probability_cloud_3d",
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


def plot_network_architecture_3d(
    model: object,
    *,
    neurons_per_layer: int = 12,
) -> go.Figure:
    """Render a sampled, interactive 3D view of a fully connected network."""
    if (
        isinstance(neurons_per_layer, bool)
        or not isinstance(neurons_per_layer, int)
        or neurons_per_layer <= 0
    ):
        raise ValueError("neurons_per_layer must be a positive integer.")

    input_dim = getattr(model, "input_dim", None)
    hidden_dims = getattr(model, "hidden_dims", None)
    output_dim = getattr(model, "output_dim", None)
    if (
        not isinstance(input_dim, int)
        or not isinstance(hidden_dims, tuple)
        or not isinstance(output_dim, int)
    ):
        raise TypeError("model must expose input_dim, hidden_dims, and output_dim.")

    dimensions = (input_dim, *hidden_dims, output_dim)
    if any(width <= 0 for width in dimensions):
        raise ValueError("model layer dimensions must be positive.")

    layer_names = [
        "Input",
        *(f"Hidden {i}" for i in range(1, len(hidden_dims) + 1)),
        "Output",
    ]
    colors = ("#527F94", "#5F8F7A", "#6675A8", "#7891A3")
    positions: list[
        tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    ] = []
    node_traces: list[go.Scatter3d] = []

    for layer_index, (name, width) in enumerate(
        zip(layer_names, dimensions, strict=True)
    ):
        visible_indices = np.linspace(
            0,
            width - 1,
            num=min(neurons_per_layer, width),
            dtype=np.intp,
        )
        angles = np.linspace(0.0, 2.0 * np.pi, len(visible_indices), endpoint=False)
        x_values = np.full(len(visible_indices), layer_index, dtype=np.float64)
        y_values = np.cos(angles)
        z_values = np.sin(angles)
        positions.append((x_values, y_values, z_values))

        node_traces.append(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="markers",
                name=f"{name} ({width})",
                text=[f"{name} neuron {index}" for index in visible_indices],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=8,
                    color=colors[layer_index % len(colors)],
                    line=dict(color="white", width=1),
                ),
            )
        )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        left_x, left_y, left_z = left
        right_x, right_y, right_z = right
        for source in range(len(left_x)):
            for target in range(len(right_x)):
                edge_x.extend((float(left_x[source]), float(right_x[target]), None))
                edge_y.extend((float(left_y[source]), float(right_y[target]), None))
                edge_z.extend((float(left_z[source]), float(right_z[target]), None))

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="rgba(82, 127, 148, 0.28)", width=1),
        hoverinfo="skip",
        name="Sampled connections",
    )

    layer_ticks = [
        f"{name}<br>{width} units"
        for name, width in zip(layer_names, dimensions, strict=True)
    ]
    figure = go.Figure(data=[edge_trace, *node_traces])
    figure.update_layout(
        title="3D network architecture (representative neurons)",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="Network layer",
                tickmode="array",
                tickvals=list(range(len(dimensions))),
                ticktext=layer_ticks,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#DCE5E9",
            ),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="manual",
            aspectratio=dict(x=max(1.6, len(dimensions) * 0.65), y=1.2, z=1.2),
            camera=dict(eye=dict(x=1.65, y=1.5, z=1.15)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=70, b=0),
        height=520,
    )
    return figure


def plot_activation_flow_animation(
    layer_activations: Sequence[ArrayLike],
    probabilities: ArrayLike,
    *,
    max_units: int = 32,
) -> go.Figure:
    """Animate sampled values as one example passes through each network layer."""
    if (
        isinstance(max_units, bool)
        or not isinstance(max_units, int)
        or max_units <= 0
    ):
        raise ValueError("max_units must be a positive integer.")
    if len(layer_activations) == 0:
        raise ValueError("layer_activations must contain the input layer.")

    stages: list[tuple[str, NDArray[np.intp], NDArray[np.floating]]] = []
    for layer_index, layer in enumerate(layer_activations):
        values = _finite_array(layer, name=f"layer_activations[{layer_index}]")
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
            raise ValueError(
                "Each activation layer must have shape (1, number_of_units)."
            )
        unit_indices = np.linspace(
            0,
            values.shape[1] - 1,
            num=min(max_units, values.shape[1]),
            dtype=np.intp,
        )
        stage_name = (
            "Input features"
            if layer_index == 0
            else f"Hidden layer {layer_index}"
        )
        stages.append((stage_name, unit_indices, values[0, unit_indices]))

    probability_values = _finite_array(probabilities, name="probabilities")
    if probability_values.ndim != 1 or probability_values.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array.")
    if np.any(probability_values < 0) or not np.isclose(
        probability_values.sum(),
        1.0,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("probabilities must be non-negative and sum to 1.")

    stages.append(
        (
            "Output probabilities",
            np.arange(len(probability_values), dtype=np.intp),
            probability_values,
        )
    )

    frame_names = [f"stage_{index}" for index in range(len(stages))]
    frames: list[go.Frame] = []
    for frame_name, (stage_name, indices, values) in zip(
        frame_names,
        stages,
        strict=True,
    ):
        if stage_name == "Output probabilities":
            labels = [f"Digit {index}" for index in indices]
        elif stage_name == "Input features":
            labels = [f"Pixel {index}" for index in indices]
        else:
            labels = [f"Unit {index}" for index in indices]
        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color="#5F8F7A",
                        hovertemplate="%{x}<br>Value: %{y:.4f}<extra></extra>",
                    )
                ],
                layout=go.Layout(
                    title_text=f"Forward pass - {stage_name}",
                    yaxis=dict(range=[0.0, max(1.0, float(np.max(values)) * 1.15)]),
                ),
            )
        )

    first_stage = stages[0]
    initial_labels = [f"Pixel {index}" for index in first_stage[1]]
    figure = go.Figure(
        data=[
            go.Bar(
                x=initial_labels,
                y=first_stage[2],
                marker_color="#527F94",
                hovertemplate="%{x}<br>Value: %{y:.4f}<extra></extra>",
            )
        ],
        frames=frames,
    )
    figure.update_layout(
        title="Forward pass - Input features",
        xaxis_title="Sampled units",
        yaxis_title="Activation / probability",
        yaxis=dict(range=[0.0, max(1.0, float(np.max(first_stage[2])) * 1.15)]),
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=70, b=90),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.14,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 850, "redraw": True},
                                "transition": {"duration": 350},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Stage: "),
                steps=[
                    dict(
                        label=stage_name,
                        method="animate",
                        args=[
                            [frame_name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for frame_name, (stage_name, _, _) in zip(
                        frame_names,
                        stages,
                        strict=True,
                    )
                ],
            )
        ],
    )
    return figure


def plot_activation_flow_animation_3d(
    layer_activations: Sequence[ArrayLike],
    probabilities: ArrayLike,
    *,
    max_units: int = 24,
    frame_duration_ms: int = 900,
) -> go.Figure:
    """Animate a single example's sampled activations through a 3D network."""
    if (
        isinstance(max_units, bool)
        or not isinstance(max_units, int)
        or max_units <= 0
    ):
        raise ValueError("max_units must be a positive integer.")
    if (
        isinstance(frame_duration_ms, bool)
        or not isinstance(frame_duration_ms, int)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer.")
    if len(layer_activations) == 0:
        raise ValueError("layer_activations must contain the input layer.")

    stages: list[tuple[str, NDArray[np.intp], NDArray[np.floating]]] = []
    for layer_index, layer in enumerate(layer_activations):
        values = _finite_array(layer, name=f"layer_activations[{layer_index}]")
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
            raise ValueError(
                "Each activation layer must have shape (1, number_of_units)."
            )
        unit_indices = np.linspace(
            0,
            values.shape[1] - 1,
            num=min(max_units, values.shape[1]),
            dtype=np.intp,
        )
        stage_name = (
            "Input features"
            if layer_index == 0
            else f"Hidden layer {layer_index}"
        )
        stages.append((stage_name, unit_indices, values[0, unit_indices]))

    probability_values = _finite_array(probabilities, name="probabilities")
    if probability_values.ndim != 1 or probability_values.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array.")
    if np.any(probability_values < 0) or not np.isclose(
        probability_values.sum(),
        1.0,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("probabilities must be non-negative and sum to 1.")

    stages.append(
        (
            "Output probabilities",
            np.arange(len(probability_values), dtype=np.intp),
            probability_values,
        )
    )

    positions: list[
        tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    ] = []
    for layer_index, (_, indices, _) in enumerate(stages):
        angles = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
        positions.append(
            (
                np.full(len(indices), layer_index, dtype=np.float64),
                np.cos(angles),
                np.sin(angles),
            )
        )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        left_x, left_y, left_z = left
        right_x, right_y, right_z = right
        for source in range(len(left_x)):
            for target in range(len(right_x)):
                edge_x.extend((float(left_x[source]), float(right_x[target]), None))
                edge_y.extend((float(left_y[source]), float(right_y[target]), None))
                edge_z.extend((float(left_z[source]), float(right_z[target]), None))

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="rgba(82, 127, 148, 0.2)", width=1),
        hoverinfo="skip",
        name="Sampled connections",
    )
    layer_names = [stage[0] for stage in stages]
    layer_traces: list[go.Scatter3d] = []
    for stage_index, (stage_name, indices, values) in enumerate(stages):
        x_values, y_values, z_values = positions[stage_index]
        maximum = float(np.max(np.abs(values)))
        normalized_values = (
            np.abs(values) / maximum
            if maximum > 0
            else np.zeros_like(values)
        )
        labels = [
            (
                f"Pixel {index}"
                if stage_index == 0
                else f"Digit {index}"
                if stage_index == len(stages) - 1
                else f"Unit {index}"
            )
            for index in indices
        ]
        layer_traces.append(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="markers",
                name=stage_name,
                text=labels,
                customdata=values,
                hovertemplate=(
                    "%{text}<br>Activation: %{customdata:.4f}<extra></extra>"
                ),
                marker=dict(
                    size=8 if stage_index != len(stages) - 1 else 11,
                    color=normalized_values
                    if stage_index == 0
                    else np.zeros(len(indices)),
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    opacity=0.95 if stage_index == 0 else 0.15,
                    showscale=stage_index == 0,
                    colorbar=(
                        dict(title="Relative<br>activation")
                        if stage_index == 0
                        else None
                    ),
                    line=dict(color="white", width=1),
                ),
            )
        )

    frame_names = [f"stage_{index}" for index in range(len(stages))]
    frames: list[go.Frame] = []
    for active_stage, frame_name in enumerate(frame_names):
        frame_data: list[go.Scatter3d] = []
        for layer_index, (_, _, values) in enumerate(stages):
            maximum = float(np.max(np.abs(values)))
            normalized_values = (
                np.abs(values) / maximum
                if maximum > 0
                else np.zeros_like(values)
            )
            frame_data.append(
                go.Scatter3d(
                    marker=dict(
                        color=normalized_values
                        if layer_index <= active_stage
                        else np.zeros(len(values)),
                        opacity=0.95 if layer_index <= active_stage else 0.15,
                    )
                )
            )
        frames.append(
            go.Frame(
                name=frame_name,
                data=frame_data,
                traces=list(range(1, len(stages) + 1)),
                layout=go.Layout(
                    title_text=f"Forward pass · {layer_names[active_stage]}"
                ),
            )
        )

    figure = go.Figure(
        data=[edge_trace, *layer_traces],
        frames=frames,
    )
    figure.update_layout(
        title=f"3D forward pass · {layer_names[0]}",
        template="plotly_white",
        scene=dict(
            xaxis=dict(
                title="Network layer",
                tickmode="array",
                tickvals=list(range(len(stages))),
                ticktext=layer_names,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#DCE5E9",
            ),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="manual",
            aspectratio=dict(x=max(1.6, len(stages) * 0.65), y=1.2, z=1.2),
            camera=dict(eye=dict(x=1.65, y=1.5, z=1.15)),
        ),
        height=580,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0,
                y=1.13,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": frame_duration_ms,
                                    "redraw": True,
                                },
                                "transition": {"duration": 350},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Stage: "),
                steps=[
                    dict(
                        label=stage_name,
                        method="animate",
                        args=[
                            [frame_name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for frame_name, stage_name in zip(
                        frame_names,
                        layer_names,
                        strict=True,
                    )
                ],
            )
        ],
    )
    return figure


def plot_pixel_surface_3d(
    image: ArrayLike,
    *,
    title: str = "3D pixel-intensity terrain",
) -> go.Figure:
    """Render a grayscale image as a 3D pixel-intensity surface."""
    pixels = _finite_array(image, name="image")
    if pixels.ndim != 2 or min(pixels.shape) == 0:
        raise ValueError("image must be a non-empty 2D array.")
    if np.any(pixels < 0) or np.any(pixels > 1):
        raise ValueError("image intensities must be in the range [0, 1].")

    figure = go.Figure(
        data=[
            go.Surface(
                x=np.arange(pixels.shape[1]),
                y=np.arange(pixels.shape[0]),
                z=pixels,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Pixel<br>intensity"),
                hovertemplate=(
                    "Column: %{x}<br>Row: %{y}"
                    "<br>Intensity: %{z:.3f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Pixel column",
            yaxis_title="Pixel row",
            zaxis_title="Normalized intensity",
            zaxis=dict(range=[0, 1]),
            bgcolor="#0B1017",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return figure


def plot_weight_surface_3d(
    weights: ArrayLike,
    *,
    title: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """Render a two-dimensional weight matrix as a signed 3D surface."""
    values = _finite_array(weights, name="weights")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("weights must be a non-empty 2D array.")

    color_limit = max(float(np.max(np.abs(values))), 1e-12)
    figure = go.Figure(
        data=[
            go.Surface(
                x=np.arange(values.shape[1]),
                y=np.arange(values.shape[0]),
                z=values,
                colorscale="RdBu",
                cmin=-color_limit,
                cmax=color_limit,
                colorbar=dict(title="Weight"),
                hovertemplate=(
                    f"{y_title}: %{{y}}<br>{x_title}: %{{x}}"
                    "<br>Weight: %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title="Signed weight",
            aspectmode="auto",
            bgcolor="#0B1017",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return figure


def plot_probability_cloud_3d(
    probabilities: ArrayLike,
    image_names: Sequence[str],
) -> go.Figure:
    """Compare every image's class probabilities in a 3D point cloud."""
    values = _finite_array(probabilities, name="probabilities")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("probabilities must be a non-empty 2D array.")
    if len(image_names) != len(values):
        raise ValueError("image_names length must match the probability rows.")
    if np.any(values < 0) or not np.allclose(
        values.sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("each probability row must be non-negative and sum to 1.")

    image_indices = np.repeat(np.arange(len(values)), values.shape[1])
    class_indices = np.tile(np.arange(values.shape[1]), len(values))
    point_probabilities = values.ravel()
    point_names = np.repeat(np.asarray(image_names, dtype=str), values.shape[1])
    figure = go.Figure(
        go.Scatter3d(
            x=image_indices,
            y=class_indices,
            z=point_probabilities,
            mode="markers",
            text=point_names,
            customdata=point_probabilities,
            marker=dict(
                size=6,
                color=point_probabilities,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Probability"),
                line=dict(color="rgba(255,255,255,0.55)", width=0.5),
            ),
            hovertemplate=(
                "%{text}<br>Digit: %{y}"
                "<br>Probability: %{customdata:.2%}<extra></extra>"
            ),
            name="Image-class probabilities",
        )
    )
    figure.update_layout(
        title="3D prediction probability cloud",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="Uploaded image",
                tickmode="array",
                tickvals=list(range(len(values))),
                ticktext=[str(index + 1) for index in range(len(values))],
                backgroundcolor="rgba(10,14,20,0.9)",
                gridcolor="#33414F",
            ),
            yaxis=dict(
                title="Digit class",
                tickmode="array",
                tickvals=list(range(values.shape[1])),
                backgroundcolor="rgba(10,14,20,0.9)",
                gridcolor="#33414F",
            ),
            zaxis=dict(
                title="Class probability",
                range=[0, 1],
                backgroundcolor="rgba(10,14,20,0.9)",
                gridcolor="#33414F",
            ),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.25)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return figure


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
                hovertemplate=(
                    "Row: %{y}<br>Column: %{x}<br>"
                    "Weight: %{z:.3f}<extra></extra>"
                ),
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
        bar_colors[true_label] = (
            "#2ca02c" if true_label == predicted_index else "#e45756"
        )
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
                marker=dict(
                    size=3,
                    color=np.arange(len(trajectory_z)),
                    colorscale="Turbo",
                ),
                line=dict(color="white", width=4),
                name="Projected trajectory",
            )
        )

    figure.update_layout(
        title="Loss landscape",
        template="plotly_dark",
        scene=dict(
            xaxis_title="Direction 1",
            yaxis_title="Direction 2",
            zaxis_title="Loss",
            bgcolor="#0B1017",
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