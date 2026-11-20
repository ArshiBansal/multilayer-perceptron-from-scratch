"""Tests for interactive model visualizations."""

from __future__ import annotations

import numpy as np

from src.model import NumpyMLP
from src.visualize import (
    plot_activation_flow_animation,
    plot_activation_flow_animation_3d,
    plot_network_architecture_3d,
    plot_pixel_surface_3d,
    plot_probability_cloud_3d,
    plot_weight_surface_3d,
)


def test_network_architecture_3d_shows_sampled_nodes_for_each_layer() -> None:
    model = NumpyMLP(
        input_dim=4,
        hidden_dims=(3,),
        output_dim=2,
        seed=3,
    )

    figure = plot_network_architecture_3d(model, neurons_per_layer=2)

    assert len(figure.data) == 4
    assert all(trace.type == "scatter3d" for trace in figure.data)
    assert [len(trace.x) for trace in figure.data[1:]] == [2, 2, 2]
    assert len(figure.data[0].x) == 2 * 2 * 2 * 3


def test_activation_flow_animation_includes_input_hidden_and_output() -> None:
    input_activations = np.array([[0.0, 0.5, 1.0, 0.2]])
    hidden_activations = np.array([[0.25, 0.75]])
    probabilities = np.array([0.1, 0.9])

    figure = plot_activation_flow_animation(
        (input_activations, hidden_activations),
        probabilities,
        max_units=3,
    )

    assert len(figure.frames) == 3
    np.testing.assert_allclose(figure.frames[0].data[0].y, [0.0, 0.5, 0.2])
    np.testing.assert_allclose(figure.frames[1].data[0].y, [0.25, 0.75])
    np.testing.assert_allclose(figure.frames[2].data[0].y, probabilities)
    assert len(figure.layout.sliders[0].steps) == 3
    assert len(figure.layout.updatemenus[0].buttons) == 2


def test_3d_activation_animation_updates_sampled_network_layers() -> None:
    model = NumpyMLP(
        input_dim=4,
        hidden_dims=(3,),
        output_dim=2,
        seed=11,
    )
    logits, cache = model.forward(np.array([[0.1, 0.4, 0.7, 0.9]]))
    probabilities = np.exp(logits[0] - np.max(logits[0]))
    probabilities /= probabilities.sum()

    figure = plot_activation_flow_animation_3d(
        cache.activations,
        probabilities,
        max_units=3,
    )

    assert len(figure.data) == 4
    assert all(trace.type == "scatter3d" for trace in figure.data)
    assert len(figure.frames) == 3
    assert all(len(frame.data) == 3 for frame in figure.frames)
    assert all(
        len(frame.traces) == 3
        and all(index > 0 for index in frame.traces)
        for frame in figure.frames
    )
    assert len(figure.layout.sliders[0].steps) == 3
    assert len(figure.layout.updatemenus[0].buttons) == 2


def test_3d_pixel_and_weight_surfaces_preserve_input_values() -> None:
    pixels = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
    weights = np.array([[-0.5, 0.25], [0.75, -1.0]], dtype=np.float32)

    pixel_figure = plot_pixel_surface_3d(pixels)
    weight_figure = plot_weight_surface_3d(
        weights,
        title="Weight surface",
        x_title="Unit",
        y_title="Feature",
    )

    assert pixel_figure.data[0].type == "surface"
    np.testing.assert_allclose(pixel_figure.data[0].z, pixels)
    assert weight_figure.data[0].type == "surface"
    np.testing.assert_allclose(weight_figure.data[0].z, weights)


def test_3d_probability_cloud_maps_each_image_to_all_digit_classes() -> None:
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )
    figure = plot_probability_cloud_3d(
        probabilities,
        ["first.png", "second.png"],
    )

    assert figure.data[0].type == "scatter3d"
    np.testing.assert_array_equal(figure.data[0].x, [0, 0, 1, 1])
    np.testing.assert_array_equal(figure.data[0].y, [0, 1, 0, 1])
    np.testing.assert_allclose(figure.data[0].z, probabilities.ravel())
    assert list(figure.data[0].text) == [
        "first.png",
        "first.png",
        "second.png",
        "second.png",
    ]
