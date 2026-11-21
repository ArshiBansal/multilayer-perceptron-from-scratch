"""Shared sidebar content for the Streamlit app."""

import streamlit as st


def render_sidebar() -> None:
    """Show concise project context below Streamlit's page navigation."""
    with st.sidebar:
        st.markdown("### MNIST MLP")
        st.caption(
            "Explore a handwritten-digit classifier built from scratch with NumPy."
        )
