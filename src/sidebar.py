"""Shared interactive workflow navigation for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

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


def render_sidebar(
    *,
    show_visual_controls: bool = False,
    show_animation_control: bool = False,
) -> SidebarSettings:
    """Show model selection, relevant 3D controls, and the portfolio link."""
    sidebar = st.sidebar
    sidebar.markdown("### 🧠 Model workspace")
    sidebar.divider()

    model_paths = sorted(MODEL_DIRECTORY.glob("*.npz"))
    if model_paths:
        model_names = [path.name for path in model_paths]
        active_name = st.session_state.get(
            "active_model_artifact",
            "mnist_mlp.npz",
        )
        if active_name not in model_names:
            active_name = model_names[0]
        selected_name = sidebar.selectbox(
            "Model artifact",
            options=model_names,
            index=model_names.index(active_name),
            key="sidebar_model_artifact",
            help=(
                "Choose the saved model used by the dashboard, prediction, "
                "explainability, and benchmark pages."
            ),
        )
        st.session_state["active_model_artifact"] = selected_name
        selected_model_path = MODEL_DIRECTORY / selected_name
    else:
        sidebar.warning("No saved model artifacts found")
        selected_model_path = None

    neurons_per_layer = int(
        st.session_state.get("active_3d_neurons_per_layer", 16)
    )
    animation_duration_ms = int(
        st.session_state.get("active_3d_animation_duration_ms", 900)
    )
    if show_visual_controls or show_animation_control:
        sidebar.markdown("**3D visual controls**")

    if show_visual_controls:
        neurons_per_layer = sidebar.slider(
            "Neurons shown per layer",
            min_value=4,
            max_value=32,
            value=neurons_per_layer,
            step=4,
            key="sidebar_3d_neurons",
            help=(
                "Adjust the sampled neuron count in the architecture and "
                "forward-pass visuals."
            ),
        )

    if show_animation_control:
        speed_label = sidebar.select_slider(
            "Forward-pass animation speed",
            options=list(ANIMATION_SPEEDS_MS),
            value=next(
                (
                    label
                    for label, duration in ANIMATION_SPEEDS_MS.items()
                    if duration == animation_duration_ms
                ),
                "Normal",
            ),
            key="sidebar_3d_animation_speed",
        )
        animation_duration_ms = ANIMATION_SPEEDS_MS[speed_label]
        st.session_state["active_3d_neurons_per_layer"] = neurons_per_layer
        st.session_state["active_3d_animation_duration_ms"] = (
            animation_duration_ms
        )

    sidebar.divider()
    sidebar.link_button("Visit my portfolio", PORTFOLIO_URL)
    return SidebarSettings(
        model_path=selected_model_path,
        neurons_per_layer=neurons_per_layer,
        animation_duration_ms=animation_duration_ms,
    )
