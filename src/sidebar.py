"""Shared model and experiment controls for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import streamlit as st

from src.model import NumpyMLP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
PORTFOLIO_URL = "https://arshibansal.github.io/Personal_Portfolio/"
ANIMATION_SPEEDS_MS = {
    "Slow": 1500,
    "Normal": 900,
    "Fast": 450,
}


@dataclass(frozen=True)
class SidebarSettings:
    """Current shared model and 3D animation controls."""

    model_path: Path | None
    neurons_per_layer: int
    animation_duration_ms: int


@st.cache_resource(show_spinner=False)
def _load_model_artifact(
    model_path: str,
    artifact_version: tuple[int, int],
) -> NumpyMLP:
    """Load and cache a model; file metadata invalidates the cache on change."""
    del artifact_version
    return NumpyMLP.load(model_path)


def _load_artifact_metadata(
    path: Path,
) -> dict[str, int | float | str | bool]:
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
            if value.ndim != 0:
                continue

            item = value.item()
            if isinstance(item, (int, float, str, bool)):
                metadata[name] = item

    return metadata


def _format_accuracy(value: object) -> str:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    ):
        return f"{value:.2%}"
    return "Not recorded"


def validation_accuracy_from_metadata(
    metadata: dict[str, int | float | str | bool],
) -> int | float | str | bool | None:
    """Return best-checkpoint accuracy, falling back to legacy metadata."""
    return metadata.get(
        "best_validation_accuracy",
        metadata.get("final_validation_accuracy"),
    )


def render_sidebar(
    *,
    show_visual_controls: bool = False,
    show_animation_control: bool = False,
) -> SidebarSettings:
    """Show model status, evaluation context, and relevant page controls."""
    sidebar = st.sidebar
    sidebar.markdown("## 🧠 MLP LAB")
    sidebar.caption("MNIST model workspace")
    sidebar.divider()

    sidebar.markdown("**ACTIVE MODEL**")
    model_paths = sorted(MODEL_DIRECTORY.glob("*.npz"))
    selected_model_path: Path | None = None

    if model_paths:
        model_names = [path.name for path in model_paths]
        preferred_name = st.session_state.get(
            "active_model_artifact",
            "mnist_mlp_256_128.npz",
        )
        if preferred_name not in model_names:
            preferred_name = model_names[0]

        widget_name = st.session_state.get("sidebar_model_artifact")
        if widget_name not in model_names:
            st.session_state["sidebar_model_artifact"] = preferred_name

        selected_name = sidebar.selectbox(
            "Model artifact",
            options=model_names,
            index=model_names.index(preferred_name),
            key="sidebar_model_artifact",
            label_visibility="collapsed",
            help="Select the model used by the app pages.",
        )
        st.session_state["active_model_artifact"] = selected_name
        selected_model_path = MODEL_DIRECTORY / selected_name

        try:
            artifact_stat = selected_model_path.stat()
            model = _load_model_artifact(
                str(selected_model_path),
                (artifact_stat.st_mtime_ns, artifact_stat.st_size),
            )
            metadata = _load_artifact_metadata(selected_model_path)
        except (BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError) as exc:
            sidebar.error(f"Model artifact could not be loaded: {exc}")
        else:
            compatible = model.input_dim == 784 and model.output_dim == 10
            if compatible:
                sidebar.success("Ready · MNIST-compatible")
            else:
                sidebar.warning(
                    "This model is not MNIST-compatible. Expected 784 inputs "
                    "and 10 output classes."
                )

            architecture = " → ".join(
                map(str, (model.input_dim, *model.hidden_dims, model.output_dim))
            )
            parameter_count = sum(
                parameter.size for parameter in model.params.values()
            )
            sidebar.caption(f"Architecture: `{architecture}`")
            sidebar.caption(f"Trainable parameters: {parameter_count:,}")

            validation_accuracy = validation_accuracy_from_metadata(metadata)
            best_epoch = metadata.get("best_epoch")
            test_accuracy = metadata.get("test_accuracy")
            seed = metadata.get("seed")
            modified = datetime.fromtimestamp(
                artifact_stat.st_mtime
            ).strftime("%Y-%m-%d %H:%M")

            sidebar.markdown("**EVALUATION PROTOCOL**")
            sidebar.caption("Dataset: MNIST · 28 × 28 grayscale")
            sidebar.caption(
                f"Validation accuracy: {_format_accuracy(validation_accuracy)}"
            )
            if isinstance(best_epoch, int) and not isinstance(best_epoch, bool):
                sidebar.caption(f"Selected checkpoint: epoch {best_epoch}")
            sidebar.caption(
                "Test set: Not evaluated"
                if test_accuracy is None
                else f"Test set: Evaluated · {_format_accuracy(test_accuracy)}"
            )
            sidebar.caption(
                f"Seed: {seed if seed is not None else 'Not recorded'}"
            )
            sidebar.caption(f"Artifact updated: {modified}")
    else:
        sidebar.warning("No model artifacts found in the models folder.")

    if show_visual_controls or show_animation_control:
        sidebar.divider()
        sidebar.markdown("**PAGE SETTINGS**")
        with sidebar.expander("Visualization controls", expanded=True):
            if show_visual_controls:
                neurons_per_layer = sidebar.slider(
                    "Neurons shown per layer",
                    min_value=4,
                    max_value=32,
                    value=int(
                        st.session_state.get("active_3d_neurons_per_layer", 16)
                    ),
                    step=4,
                    key="sidebar_3d_neurons",
                    help=(
                        "Adjust the sampled neuron count in architecture and "
                        "forward-pass views."
                    ),
                )
                st.session_state["active_3d_neurons_per_layer"] = (
                    neurons_per_layer
                )
            else:
                neurons_per_layer = int(
                    st.session_state.get("active_3d_neurons_per_layer", 16)
                )

            if show_animation_control:
                current_duration = int(
                    st.session_state.get("active_3d_animation_duration_ms", 900)
                )
                current_speed = next(
                    (
                        label
                        for label, duration in ANIMATION_SPEEDS_MS.items()
                        if duration == current_duration
                    ),
                    "Normal",
                )
                speed_label = sidebar.select_slider(
                    "Forward-pass animation speed",
                    options=list(ANIMATION_SPEEDS_MS),
                    value=current_speed,
                    key="sidebar_3d_animation_speed",
                )
                animation_duration_ms = ANIMATION_SPEEDS_MS[speed_label]
                st.session_state["active_3d_animation_duration_ms"] = (
                    animation_duration_ms
                )
            else:
                animation_duration_ms = int(
                    st.session_state.get("active_3d_animation_duration_ms", 900)
                )

            if sidebar.button("Reset visualization controls"):
                st.session_state["active_3d_neurons_per_layer"] = 16
                st.session_state["active_3d_animation_duration_ms"] = 900
                st.session_state["sidebar_3d_neurons"] = 16
                st.session_state["sidebar_3d_animation_speed"] = "Normal"
                st.rerun()
    else:
        neurons_per_layer = int(
            st.session_state.get("active_3d_neurons_per_layer", 16)
        )
        animation_duration_ms = int(
            st.session_state.get("active_3d_animation_duration_ms", 900)
        )

    sidebar.divider()
    sidebar.link_button("About this project", PORTFOLIO_URL)

    return SidebarSettings(
        model_path=selected_model_path,
        neurons_per_layer=neurons_per_layer,
        animation_duration_ms=animation_duration_ms,
    )