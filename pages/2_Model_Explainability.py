"""Explore the saved MLP architecture and its learned parameters."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st

from src.artifacts import (
    loss_landscape_path,
    validate_loss_landscape_model,
)
from src.model import NumpyMLP
from src.sidebar import render_sidebar, validation_accuracy_from_metadata
from src.visualize import (
    plot_loss_landscape,
    plot_network_architecture_3d,
    plot_weight_surface_3d,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str, artifact_version: tuple[int, int]) -> NumpyMLP:
    """Cache the saved model for this Streamlit process."""
    del artifact_version
    return NumpyMLP.load(model_path)


def load_artifact_metadata(path: Path) -> dict[str, int | float | str | bool]:
    """Read scalar metadata without enabling pickle loading."""
    metadata: dict[str, int | float | str | bool] = {}
    with np.load(path, allow_pickle=False) as artifact:
        for key in artifact.files:
            if key.startswith("metadata_"):
                name = key.removeprefix("metadata_")
            elif key in {
                "test_accuracy",
                "final_validation_accuracy",
                "best_validation_accuracy",
                "best_validation_loss",
                "best_epoch",
            }:
                name = key
            else:
                continue

            value = artifact[key]
            if value.ndim == 0:
                item = value.item()
                if isinstance(item, (int, float, str, bool)):
                    metadata[name] = item
    return metadata


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "Not recorded"


st.set_page_config(
    page_title="Model Explainability",
    page_icon="🧭",
    layout="wide",
)
sidebar_settings = render_sidebar(show_visual_controls=True)
if sidebar_settings.model_path is not None:
    MODEL_PATH = sidebar_settings.model_path
LOSS_LANDSCAPE_PATH = loss_landscape_path(MODEL_PATH, REPORT_DIR)

st.title("Model Explainability")
st.markdown(
    "Explore **what the network contains and how its learned parameters are "
    "organized**. Visualizations describe the saved model; weight patterns "
    "are not, by themselves, causal explanations of a prediction."
)

if not MODEL_PATH.is_file():
    st.error(f"Trained model not found: `{MODEL_PATH}`")
    st.info("Train the model first, for example with `python scripts/train.py`.")
    st.stop()

try:
    artifact_stat = MODEL_PATH.stat()
    model = load_model(
        str(MODEL_PATH),
        (artifact_stat.st_mtime_ns, artifact_stat.st_size),
    )
    metadata = load_artifact_metadata(MODEL_PATH)
except (EOFError, KeyError, OSError, TypeError, ValueError) as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

layer_nodes: list[tuple[str, int, str]] = [
    ("Input", model.input_dim, "normalized pixels"),
]
layer_nodes.extend(
    (f"Hidden {index}", width, model.activation)
    for index, width in enumerate(model.hidden_dims, start=1)
)
layer_nodes.append(("Output", model.output_dim, "softmax digit scores"))

node_colors = ("#527F94", "#5F8F7A", "#6675A8")
node_markup: list[str] = []
for index, (label, width, role) in enumerate(layer_nodes):
    if index:
        weights = model.params[f"W{index}"]
        biases = model.params[f"b{index}"]
        connection_parameters = weights.size + biases.size
        node_markup.append(
            '<div style="color:#9CC8C4;text-align:center;padding:0 4px">'
            f'→<br><small>{connection_parameters:,} params</small></div>'
        )

    color = node_colors[index % len(node_colors)]
    node_markup.append(
        f'<div style="flex:1;min-width:120px;padding:14px;'
        f'border:1px solid {color};border-radius:10px;'
        f'background:rgba(11,16,23,.82);text-align:center">'
        f'<small style="color:#9CC8C4">{escape(label.upper())}</small><br>'
        f'<strong style="font-size:1.55rem">{width:,}</strong><br>'
        f'<small>{escape(role)}</small></div>'
    )

test_accuracy = metadata.get("test_accuracy")
st.subheader("Neural X-ray")
st.caption(
    "Follow the model from normalized pixels through each learned layer to "
    "the digit scores. Connection labels show the parameters learned at that step."
)
with st.container(border=True):
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'gap:8px;flex-wrap:wrap;padding:10px 0">'
        + "".join(node_markup)
        + "</div>",
        unsafe_allow_html=True,
    )
    evaluation_columns = st.columns(2)
    evaluation_columns[0].caption(
        "Validation · "
        f"{format_metric(validation_accuracy_from_metadata(metadata))}"
    )
    evaluation_columns[1].caption(
        "Official test set · "
        + (
            "not evaluated"
            if test_accuracy is None
            else f"evaluated · {format_metric(test_accuracy)}"
        )
    )

architecture_tab, weights_tab, landscape_tab = st.tabs(
    ["Architecture", "Learned weights", "Loss landscape"]
)

with architecture_tab:
    st.subheader("Trace the network structure")
    architecture_columns = st.columns((1.5, 1))
    with architecture_columns[0]:
        st.plotly_chart(
            plot_network_architecture_3d(
                model,
                neurons_per_layer=sidebar_settings.neurons_per_layer,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.caption(
            "**What:** representative neurons and dense feed-forward links by "
            "layer. **Ask:** how does information travel from pixels to digits?"
        )
    with architecture_columns[1]:
        st.markdown("#### Forward pass")
        st.markdown(
            """
            1. **Input:** each 28 × 28 image becomes 784 normalized pixel values.
            2. **Hidden layers:** learned affine transforms are followed by the
               configured nonlinear activation.
            3. **Output:** ten logits are converted to probabilities for digits
               0–9.
            """
        )
        st.info(
            "The 3D view samples representative neurons and connections so the "
            "full architecture remains readable."
        )

    layer_rows = []
    for layer_index in range(1, model.layer_count + 1):
        weights = model.params[f"W{layer_index}"]
        biases = model.params[f"b{layer_index}"]
        layer_rows.append(
            {
                "Layer": layer_index,
                "Role": (
                    "Output"
                    if layer_index == model.layer_count
                    else "Hidden"
                ),
                "Weights": f"{weights.shape[0]} × {weights.shape[1]}",
                "Biases": len(biases),
                "Parameters": weights.size + biases.size,
                "Weight mean": float(weights.mean()),
                "Weight standard deviation": float(weights.std()),
            }
        )

    with st.expander("Inspect layer dimensions and parameter statistics"):
        st.dataframe(layer_rows, use_container_width=True, hide_index=True)

with weights_tab:
    st.subheader("Inspect learned weight patterns")
    st.caption(
        "These views summarize the trained parameters. First-layer patterns "
        "can resemble strokes, but should not be read as pixel-level attribution."
    )

    surface_tab, class_tab = st.tabs(
        ["3D input-weight surface", "3D class weights"]
    )

    with surface_tab:
        selected_unit = st.slider(
            "First-layer neuron",
            min_value=1,
            max_value=model.hidden_dims[0],
            value=1,
            step=1,
            key="weight_surface_unit",
        )
        first_layer_weights = model.params["W1"][:, selected_unit - 1]
        if first_layer_weights.size != 28 * 28:
            st.info(
                "The 3D pixel-weight surface is available for models with "
                "784 input features."
            )
        else:
            st.plotly_chart(
                plot_weight_surface_3d(
                    first_layer_weights.reshape(28, 28),
                    title=f"3D input-weight surface · neuron {selected_unit}",
                    x_title="Pixel column",
                    y_title="Pixel row",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "**What:** signed input weights for one hidden neuron, mapped "
                "onto pixel locations. **Ask:** which regions excite or "
                "suppress this feature?"
            )

    with class_tab:
        output_weights = model.params[f"W{model.layer_count}"].T
        st.caption(
            "**What:** signed output-layer weights connect hidden units to "
            "digit classes. **Ask:** which hidden signals support or suppress "
            "each digit?"
        )
        st.plotly_chart(
            plot_weight_surface_3d(
                output_weights,
                title="3D output-weight surface",
                x_title="Final hidden unit",
                y_title="Output digit",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

with landscape_tab:
    st.subheader("Explore an optimization slice")
    st.caption(
        "This visualization shows a two-dimensional slice through selected "
        "output-layer parameters, not the full high-dimensional loss surface."
    )

    if not LOSS_LANDSCAPE_PATH.is_file():
        st.info(
            f"No loss-landscape data is available for `{MODEL_PATH.name}`. "
            "Choose that model in `notebooks/04_3D_Loss_Landscape.ipynb` and "
            "run all cells to generate its matching landscape."
        )
    else:
        try:
            with np.load(LOSS_LANDSCAPE_PATH, allow_pickle=False) as landscape:
                required = {
                    "x_grid",
                    "y_grid",
                    "loss_grid",
                    "trajectory_coordinates",
                    "trajectory_probe_losses",
                }
                missing = required.difference(landscape.files)
                if missing:
                    raise ValueError(
                        f"Landscape artifact is missing keys: {sorted(missing)}."
                    )
                landscape_model = (
                    str(landscape["model_artifact"].item())
                    if "model_artifact" in landscape.files
                    else None
                )
                validate_loss_landscape_model(MODEL_PATH, landscape_model)

                figure = plot_loss_landscape(
                    landscape["x_grid"],
                    landscape["y_grid"],
                    landscape["loss_grid"],
                    trajectory_coordinates=landscape["trajectory_coordinates"],
                    trajectory_losses=landscape["trajectory_probe_losses"],
                )
                explained_variance = landscape.get("explained_variance")

            if explained_variance is not None:
                st.metric(
                    "Variance covered by plotted directions",
                    f"{float(np.sum(explained_variance)):.1%}",
                )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "**What:** loss across two selected parameter directions, with "
                "the recorded training path. **Ask:** does the path move toward "
                "a lower-loss region?"
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            st.error(f"Could not load the loss-landscape artifact: {exc}")