"""Home page for the MNIST MLP Streamlit app."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.model import NumpyMLP
from src.sidebar import render_sidebar

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str, artifact_version: tuple[int, int]) -> NumpyMLP:
    del artifact_version
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
sidebar_settings = render_sidebar()
if sidebar_settings.model_path is not None:
    MODEL_PATH = sidebar_settings.model_path

st.title("MNIST Multi-Layer Perceptron")
st.markdown(
    """
    A focused workspace for exploring a **fully connected neural network
    implemented from scratch with NumPy**. Predict handwritten digits, inspect
    learned parameters, and compare reproducible training experiments.
    """
)
st.info(
    "This dashboard loads the saved model artifact. Dataset download and model "
    "training are deliberate, separate workflows."
)

if MODEL_PATH.is_file():
    try:
        artifact_stat = MODEL_PATH.stat()
        model = load_model(
            str(MODEL_PATH),
            (artifact_stat.st_mtime_ns, artifact_stat.st_size),
        )
        metadata = load_model_metadata(MODEL_PATH)
    except (EOFError, KeyError, OSError, TypeError, ValueError) as exc:
        st.error(f"Could not load the model artifact: {exc}")
        st.caption(f"Expected artifact: `{MODEL_PATH}`")
    else:
        parameter_count = sum(parameter.size for parameter in model.params.values())
        architecture_label = " → ".join(
            map(str, (model.input_dim, *model.hidden_dims, model.output_dim))
        )
        test_accuracy = metadata.get("test_accuracy")
        test_status = (
            "Not evaluated"
            if test_accuracy is None
            else f"Evaluated · {format_accuracy(test_accuracy)}"
        )
        seed_label = metadata.get("seed", "Not recorded")
        updated_label = datetime.fromtimestamp(artifact_stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )

        st.subheader("Model dossier")
        with st.container(border=True):
            dossier_model, dossier_record = st.columns((1.4, 1))
            with dossier_model:
                st.caption("MODEL SIGNATURE · MNIST")
                st.markdown(f"### `{architecture_label}`")
                st.write(
                    f"{model.activation.upper()} hidden layers · "
                    f"{model.initialization.upper()} initialization"
                )
                st.caption(
                    f"{parameter_count:,} trainable parameters · "
                    "28 × 28 grayscale input · 10 digit classes"
                )
            with dossier_record:
                st.caption("EVALUATION TRAIL")
                st.write(
                    "**Validation** · "
                    f"{format_accuracy(metadata.get('final_validation_accuracy'))}"
                )
                st.write(f"**Official test set** · {test_status}")
                st.caption(
                    f"Seed {seed_label} · Artifact updated {updated_label}"
                )
else:
    st.warning(f"No trained model found at `{MODEL_PATH}`.")
    st.code("python scripts/train.py", language="powershell")

st.divider()
st.subheader("Choose a workflow")
st.write(
    "Each page is designed around one task, from preparing inputs to "
    "interpreting a model or comparing experiments."
)
page_columns = st.columns(4)

with page_columns[0]:
    st.markdown("#### 01 · Prediction Studio")
    st.write(
        "Predict single images or batches, review image quality, and inspect "
        "class probabilities."
    )
    st.page_link(
        "pages/1_Prediction_Studio.py",
        label="Open Prediction Studio",
        icon="🔢",
    )

with page_columns[1]:
    st.markdown("#### 02 · Model Explainability")
    st.write(
        "Explore network structure, learned weights, and an optimization "
        "landscape."
    )
    st.page_link(
        "pages/2_Model_Explainability.py",
        label="Open Model Explainability",
        icon="🧭",
    )

with page_columns[2]:
    st.markdown("#### 03 · Training & Experiments")
    st.write(
        "Configure reproducible CLI runs and compare validation results across "
        "experiments."
    )
    st.page_link(
        "pages/3_Training_Experiments.py",
        label="Open Training & Experiments",
        icon="🧪",
    )

with page_columns[3]:
    st.markdown("#### 04 · Model Benchmark")
    st.write(
        "Measure inference latency, throughput, compute, and model footprint."
    )
    st.page_link(
        "pages/4_Model_Benchmark.py",
        label="Open Model Benchmark",
        icon="⏱️",
    )

st.divider()
workflow_columns = st.columns(3)
workflow_columns[0].markdown("**Prepare**  \nNormalize and center digit images.")
workflow_columns[1].markdown(
    "**Predict**  \nRun inference through the saved NumPy MLP."
)
workflow_columns[2].markdown(
    "**Improve**  \nCompare training runs using validation metrics."
)
st.caption(
    "For dataset setup and training dependencies, see `scripts/train.py` and "
    "the project README."
)