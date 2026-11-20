"""Home page for the MNIST MLP Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mnist_mlp.model import NumpyMLP


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str) -> NumpyMLP:
    return NumpyMLP.load(model_path)


def load_model_metadata(path: Path) -> dict[str, int | float | str | bool]:
    """Read scalar metadata from either the current or notebook artifact format."""
    import numpy as np

    metadata: dict[str, int | float | str | bool] = {}
    with np.load(path, allow_pickle=False) as artifact:
        for key in artifact.files:
            if key.startswith("metadata_"):
                name = key.removeprefix("metadata_")
            elif key in {"test_accuracy", "final_validation_accuracy"}:
                name = key
            else:
                continue

            value = artifact[key]
            if value.ndim == 0:
                item = value.item()
                if isinstance(item, (int, float, str, bool)):
                    metadata[name] = item

    return metadata


def format_accuracy(value: object) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else "Not recorded"


st.set_page_config(
    page_title="MNIST MLP — From Scratch",
    page_icon="🧠",
    layout="wide",
)

st.title("MNIST Multi-Layer Perceptron")
st.markdown(
    """
    An interactive showcase of a **fully connected neural network implemented
    from scratch with NumPy**. Explore digit predictions, model weights, and
    training experiments.
    """
)

st.info(
    "The deployed app loads a saved model artifact. Training and MNIST "
    "downloading are separate tasks and are not run when the app starts."
)

if MODEL_PATH.is_file():
    try:
        model = load_model(str(MODEL_PATH))
        metadata = load_model_metadata(MODEL_PATH)
    except (OSError, TypeError, ValueError) as exc:
        st.error(f"Could not load the model artifact: {exc}")
        st.caption(f"Expected artifact: `{MODEL_PATH}`")
    else:
        parameter_count = sum(parameter.size for parameter in model.params.values())
        columns = st.columns(4)
        columns[0].metric(
            "Architecture",
            " -> ".join(
                map(str, (model.input_dim, *model.hidden_dims, model.output_dim))
            ),
        )
        columns[1].metric("Parameters", f"{parameter_count:,}")
        columns[2].metric(
            "Validation accuracy",
            format_accuracy(metadata.get("final_validation_accuracy")),
        )
        columns[3].metric(
            "Test accuracy",
            format_accuracy(metadata.get("test_accuracy")),
        )
else:
    st.warning(f"No trained model found at `{MODEL_PATH}`.")
    st.code("python scripts/train.py", language="powershell")

st.subheader("Explore the project")
page_columns = st.columns(3)

with page_columns[0]:
    st.markdown("#### Predict a digit")
    st.write("Upload a handwritten digit and inspect the model's class probabilities.")
    st.page_link(
        "pages/1_Predict_Digit.py",
        label="Open digit prediction",
        icon="🔢",
    )

with page_columns[1]:
    st.markdown("#### Model insights")
    st.write("Explore learned weights, parameter statistics, and the loss landscape.")
    st.page_link(
        "pages/2_Model_Insights.py",
        label="Open model insights",
        icon="📊",
    )

with page_columns[2]:
    st.markdown("#### Experiments")
    st.write("Configure training runs and compare exported validation results.")
    st.page_link(
        "pages/3_Experiments.py",
        label="Open experiments",
        icon="🧪",
    )

st.divider()
st.subheader("How the model works")
st.markdown(
    """
    1. Flatten each 28 × 28 grayscale image into 784 normalized pixel features.
    2. Apply one or more fully connected hidden layers with a nonlinear activation.
    3. Produce logits for the ten digit classes and convert them to probabilities.
    4. Train with NumPy implementations of cross-entropy, backpropagation, and SGD.
    """
)
st.caption(
    "For training and dataset setup, install the training dependencies and see "
    "`scripts/train.py` and the notebooks."
)