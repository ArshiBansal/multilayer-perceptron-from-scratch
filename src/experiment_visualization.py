"""Visualization helpers for comparing MNIST experiment results."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def get_parameter_count(row: dict[str, Any]) -> int | None:
    """Return recorded parameters or estimate them from an MNIST MLP width list."""
    if "parameters" in row:
        count = int(row["parameters"])
        return count if count > 0 else None
    if "hidden_dims" not in row:
        return None

    try:
        hidden_dims = tuple(
            int(width.strip()) for width in row["hidden_dims"].split(",")
        )
    except (AttributeError, ValueError):
        return None
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        return None

    dimensions = (784, *hidden_dims, 10)
    return sum(
        (fan_in + 1) * fan_out
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True)
    )


def make_experiment_tradeoff_3d(
    rows: list[dict[str, Any]],
    *,
    depth_metric: str,
) -> go.Figure:
    """Plot model complexity or runtime alongside validation metrics in 3D."""
    if depth_metric not in {"parameters", "seconds"}:
        raise ValueError("depth_metric must be 'parameters' or 'seconds'.")

    def depth_value(row: dict[str, Any]) -> float | int:
        if depth_metric == "parameters":
            parameter_count = get_parameter_count(row)
            if parameter_count is None:
                raise ValueError("3D parameter plots require parameter counts.")
            return parameter_count
        return float(row["seconds"])

    depth_label = (
        "Trainable parameters"
        if depth_metric == "parameters"
        else "Training time (seconds)"
    )
    plot_title = (
        "3D model-complexity and validation landscape"
        if depth_metric == "parameters"
        else "3D training-time and validation landscape"
    )
    figure = go.Figure()
    groups = list(dict.fromkeys(row["group"] for row in rows))

    for group in groups:
        group_rows = [
            row
            for row in rows
            if row["group"] == group
            and (
                get_parameter_count(row) is not None
                if depth_metric == "parameters"
                else row.get("seconds") is not None
            )
        ]
        if not group_rows:
            continue
        figure.add_trace(
            go.Scatter3d(
                name=group,
                mode="markers+text",
                x=[depth_value(row) for row in group_rows],
                y=[row["val_loss"] for row in group_rows],
                z=[row["val_accuracy"] for row in group_rows],
                text=[row["name"] for row in group_rows],
                marker=dict(size=12, opacity=0.8),
                customdata=[
                    [
                        row.get("hidden_dims", "Not recorded"),
                        row.get("seconds", "Not recorded"),
                    ]
                    for row in group_rows
                ],
                hovertemplate=(
                    f"%{{text}}<br>{depth_label}: %{{x:,.2f}}"
                    "<br>Validation loss: %{y:.4f}"
                    "<br>Validation accuracy: %{z:.2%}"
                    "<br>Architecture: %{customdata[0]}"
                    "<br>Runtime: %{customdata[1]} s<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=plot_title,
        template="plotly_dark",
        scene=dict(
            xaxis_title=depth_label,
            yaxis_title="Validation loss",
            zaxis_title="Validation accuracy",
            zaxis=dict(tickformat=".0%", range=[0, 1]),
            bgcolor="#0B1017",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        legend_title="Experiment group",
        height=600,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return figure
