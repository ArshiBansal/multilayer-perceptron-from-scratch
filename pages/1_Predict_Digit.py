"""Streamlit page for predicting a digit from an uploaded image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from mnist_mlp.model import NumpyMLP
from mnist_mlp.visualize import plot_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MNIST_CANVAS_SIZE = 28
DIGIT_SIZE = 20


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str) -> NumpyMLP:
    """Load and cache the trained model for this Streamlit process."""
    return NumpyMLP.load(model_path)


def prepare_digit(
    image_bytes: bytes,
    *,
    invert: bool,
) -> tuple[Image.Image, np.ndarray]:
    """Convert an uploaded image into a centered MNIST-style digit.

    The foreground is cropped, scaled to fit within a 20 × 20 area, and centered
    on a black 28 × 28 canvas. The returned array uses the model's normalized
    [0, 1] pixel range.
    """
    with Image.open(image_bytes) as source:
        grayscale = source.convert("L")

    if invert:
        grayscale = ImageOps.invert(grayscale)

    # MNIST digits are bright on a dark background.
    foreground = grayscale.point(lambda pixel: 255 if pixel > 30 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise ValueError("No visible digit found. Try a clearer image.")

    cropped = grayscale.crop(bounds)
    width, height = cropped.size
    if width == 0 or height == 0:
        raise ValueError("The uploaded image does not contain a usable digit.")

    scale = DIGIT_SIZE / max(width, height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE), color=0)
    offset = (
        (MNIST_CANVAS_SIZE - resized.width) // 2,
        (MNIST_CANVAS_SIZE - resized.height) // 2,
    )
    canvas.paste(resized, offset)

    normalized = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    return canvas, normalized


st.set_page_config(
    page_title="Predict a digit",
    page_icon="🔢",
    layout="centered",
)

st.title("Predict a handwritten digit")
st.write(
    "Upload an image containing one digit. For best results, use a centered "
    "digit with clear contrast against its background."
)

if not MODEL_PATH.is_file():
    st.error(f"Trained model not found: `{MODEL_PATH}`")
    st.info("Train the model first, for example with `python scripts/train.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

with st.form("digit_prediction_form"):
    uploaded_file = st.file_uploader(
        "Choose a digit image",
        type=("png", "jpg", "jpeg", "bmp"),
        help="Images are converted to grayscale and centered on a 28 × 28 canvas.",
    )
    invert = st.checkbox(
        "Invert image",
        value=False,
        help="Enable this for dark ink on a light background.",
    )
    submitted = st.form_submit_button("Predict digit", type="primary")

if submitted:
    if uploaded_file is None:
        st.warning("Upload an image before requesting a prediction.")
        st.stop()

    image_bytes = uploaded_file.getvalue()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.error("Image is too large. Please upload a file smaller than 5 MB.")
        st.stop()

    try:
        prepared_image, normalized_image = prepare_digit(
            image_bytes,
            invert=invert,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process the uploaded image: {exc}")
        st.stop()

    probabilities = model.predict_proba(
        normalized_image.reshape(1, -1)
    )[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    left, right = st.columns((1, 2))
    with left:
        st.image(
            prepared_image,
            caption="Model input (28 × 28)",
            width=180,
        )
    with right:
        st.metric("Predicted digit", predicted_digit)
        st.metric("Model probability", f"{confidence:.1%}")

    figure = plot_prediction(
        np.asarray(prepared_image),
        probabilities,
    )
    st.plotly_chart(figure, use_container_width=True)

    top_indices = np.argsort(probabilities)[::-1][:3]
    st.caption(
        "Top predictions: "
        + " · ".join(
            f"{digit} ({probabilities[digit]:.1%})"
            for digit in top_indices
        )
    )

st.caption(
    "Predictions depend on image quality and similarity to MNIST handwriting. "
    "The displayed probabilities are model outputs, not calibrated confidence estimates."
)