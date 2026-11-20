"""Streamlit page for inspecting the saved MNIST MLP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mnist_mlp.model import NumpyMLP
from mnist_mlp.visualize import plot_first_layer_weights, plot_loss_landscape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"
LOSS_LANDSCAPE_PATH = PROJECT_ROOT / "reports" / "figures" / "loss_surface.npz"


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str) -> NumpyMLP:
    return NumpyMLP.load(model_path)


def load_artifact_metadata(path: Path) -> dict[str, int | float | str | bool]:
    """Read scalar metadata without enabling pickle loading."""
    metadata: dict[str, int | float | str | bool] = {}

    with np.load(path, allow_pickle=False) as artifact:
        for key in artifact.files:
            if key.startswith("metadata_"):
                name = key.removeprefix("metadata_")
            elif key in {"test_accuracy", "final_validation_accuracy"}:
                # Also support artifacts written by the original notebook.
                name = key
            else:
                continue

            value = artifact[key]
            if value.ndim != 0:
                continue

            item = value.item()
            if isinstance(item, (int, float, str, bool)):
                metadata[name] = item

    return metadata


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "Not recorded"


st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide",
)

st.title("Model Insights")
st.write(
    "Inspect the saved model's architecture, learned weights, and optional "
    "loss-landscape visualization."
)

if not MODEL_PATH.is_file():
    st.error(f"Trained model not found: `{MODEL_PATH}`")
    st.info("Train the model first, for example with `python scripts/train.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    metadata = load_artifact_metadata(MODEL_PATH)
except (OSError, TypeError, ValueError) as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

parameter_count = sum(parameter.size for parameter in model.params.values())
test_accuracy = metadata.get("test_accuracy")
validation_accuracy = metadata.get("final_validation_accuracy")

metric_columns = st.columns(4)
metric_columns[0].metric("Architecture", " → ".join(
    map(str, (model.input_dim, *model.hidden_dims, model.output_dim))
))
metric_columns[1].metric("Parameters", f"{parameter_count:,}")
metric_columns[2].metric("Validation accuracy", format_metric(validation_accuracy))
metric_columns[3].metric("Test accuracy", format_metric(test_accuracy))

if test_accuracy is None:
    st.info(
        "Test accuracy is not stored in this artifact. To record it, run "
        "`python scripts/train.py --evaluate-test` after model selection."
    )

st.subheader("Parameter summary")

parameter_rows = []
for layer_index in range(1, model.layer_count + 1):
    weights = model.params[f"W{layer_index}"]
    biases = model.params[f"b{layer_index}"]
    parameter_rows.append(
        {
            "Layer": layer_index,
            "Weights shape": str(weights.shape),
            "Biases shape": str(biases.shape),
            "Parameters": weights.size + biases.size,
            "Weight mean": float(weights.mean()),
            "Weight std": float(weights.std()),
        }
    )

st.dataframe(
    parameter_rows,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Parameter distributions")

distribution_figure = go.Figure()
for layer_index in range(1, model.layer_count + 1):
    weights = model.params[f"W{layer_index}"]
    distribution_figure.add_trace(
        go.Histogram(
            x=weights.ravel(),
            nbinsx=60,
            name=f"Layer {layer_index}",
            opacity=0.65,
        )
    )

distribution_figure.update_layout(
    title="Weight-value distributions by layer",
    xaxis_title="Weight value",
    yaxis_title="Count",
    barmode="overlay",
)
st.plotly_chart(distribution_figure, use_container_width=True)

st.subheader("First-layer weight patterns")
st.caption(
    "Each panel shows one input-to-hidden-unit weight vector reshaped to 28 × 28. "
    "These patterns are not guaranteed to have a simple visual interpretation."
)

units_to_show = st.slider(
    "Hidden units to display",
    min_value=1,
    max_value=model.hidden_dims[0],
    value=min(32, model.hidden_dims[0]),
    step=1,
)
st.plotly_chart(
    plot_first_layer_weights(model, max_units=units_to_show),
    use_container_width=True,
)

st.subheader("Output-layer weights")
st.caption(
    "Rows represent output classes; columns represent units in the final hidden layer."
)

output_weights = model.params[f"W{model.layer_count}"].T
output_figure = go.Figure(
    go.Heatmap(
        z=output_weights,
        x=[f"Unit {index}" for index in range(output_weights.shape[1])],
        y=[str(index) for index in range(model.output_dim)],
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Weight"),
        hovertemplate=(
            "Class: %{y}<br>Hidden unit: %{x}<br>Weight: %{z:.4f}<extra></extra>"
        ),
    )
)
output_figure.update_layout(
    title="Output weights by digit class",
    xaxis_title="Hidden unit",
    yaxis_title="Digit class",
)
st.plotly_chart(output_figure, use_container_width=True)

st.subheader("3D loss landscape")

if not LOSS_LANDSCAPE_PATH.is_file():
    st.info(
        "No saved loss-landscape data found. Run "
        "`notebooks/04_3D_Loss_Landscape.ipynb` to generate it."
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

            figure = plot_loss_landscape(
                landscape["x_grid"],
                landscape["y_grid"],
                landscape["loss_grid"],
                trajectory_coordinates=landscape["trajectory_coordinates"],
                trajectory_losses=landscape["trajectory_probe_losses"],
            )

            explained_variance = landscape.get("explained_variance")

        if explained_variance is not None:
            st.caption(
                "Variance explained by plotted directions: "
                f"{float(np.sum(explained_variance)):.1%}"
            )

        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "This is a two-dimensional output-layer slice, not the full network "
            "loss landscape."
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        st.error(f"Could not load the loss-landscape artifact: {exc}")