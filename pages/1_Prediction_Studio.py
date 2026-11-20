"""Streamlit page for predicting a digit from an uploaded image."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.image_processing import (
    BLUR_SCORE_THRESHOLD,
    SUPPORTED_IMAGE_EXTENSIONS,
    DigitPreparation,
    has_supported_image_extension,
    prepare_digit,
)
from src.model import NumpyMLP, softmax
from src.sidebar import render_sidebar
from src.visualize import (
    plot_activation_flow_animation_3d,
    plot_pixel_surface_3d,
    plot_probability_cloud_3d,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_FILES = 20
SUPPORTED_FORMAT_LABEL = ", ".join(
    extension.removeprefix(".").upper()
    for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS)
)
SUPPORTED_UPLOAD_TYPES = tuple(
    extension.removeprefix(".")
    for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS)
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str, artifact_version: tuple[int, int]) -> NumpyMLP:
    """Load and cache the trained model for this Streamlit process."""
    del artifact_version
    return NumpyMLP.load(model_path)


st.set_page_config(
    page_title="Prediction Studio",
    page_icon="🔢",
    layout="wide",
)
sidebar_settings = render_sidebar(
    show_visual_controls=True,
    show_animation_control=True,
)
if sidebar_settings.model_path is not None:
    MODEL_PATH = sidebar_settings.model_path

st.title("Prediction Studio")
st.write(
    "Predict one digit or compare a batch. Review confidence patterns, export "
    "results, and animate how a selected image moves through the network."
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
except (EOFError, KeyError, OSError, TypeError, ValueError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.caption(f"Expected artifact: `{MODEL_PATH}`")
    st.stop()

with st.form("digit_prediction_form"):
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=(
            f"Supported formats: {SUPPORTED_FORMAT_LABEL}. Upload up to "
            f"{MAX_BATCH_FILES} images, each limited to 5 MB. Only the first "
            "frame is used for animated images."
        ),
    )
    invert = st.checkbox(
        "Invert image",
        value=False,
        help="Enable this for dark ink on a light background.",
    )
    auto_sharpen = st.checkbox(
        "Auto-sharpen images flagged as blurry",
        value=True,
        help=(
            "A gentle unsharp mask is applied only when the edge-sharpness "
            "heuristic detects soft edges."
        ),
    )
    sharpening_amount = st.slider(
        "Sharpening amount",
        min_value=0,
        max_value=200,
        value=125,
        step=25,
        disabled=not auto_sharpen,
        help="Higher values emphasize edges; sharpening cannot restore lost detail.",
    )
    submitted = st.form_submit_button("Predict digit", type="primary")

if submitted:
    if not uploaded_files:
        st.session_state.pop("digit_batch_results", None)
        st.warning("Upload at least one image before requesting predictions.")
        st.stop()

    if len(uploaded_files) > MAX_BATCH_FILES:
        st.session_state.pop("digit_batch_results", None)
        st.error(f"Upload no more than {MAX_BATCH_FILES} images at a time.")
        st.stop()

    digit_preparations: list[DigitPreparation] = []
    file_names: list[str] = []
    failed_uploads: list[tuple[str, str]] = []
    progress = st.progress(0, text="Preparing uploaded images...")

    for file_index, uploaded_file in enumerate(uploaded_files):
        if not has_supported_image_extension(uploaded_file.name):
            failed_uploads.append(
                (
                    uploaded_file.name,
                    f"Unsupported file extension. Supported formats: "
                    f"{SUPPORTED_FORMAT_LABEL}.",
                )
            )
        elif uploaded_file.size > MAX_UPLOAD_BYTES:
            failed_uploads.append(
                (uploaded_file.name, "File exceeds the 5 MB per-image limit.")
            )
        else:
            try:
                preparation = prepare_digit(
                    uploaded_file.getvalue(),
                    invert=invert,
                    auto_sharpen=auto_sharpen,
                    sharpening_amount=sharpening_amount,
                )
            except (
                Image.DecompressionBombError,
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as exc:
                failed_uploads.append((uploaded_file.name, str(exc)))
            else:
                digit_preparations.append(preparation)
                file_names.append(uploaded_file.name)

        progress.progress(
            (file_index + 1) / len(uploaded_files),
            text=f"Processed {file_index + 1} of {len(uploaded_files)} files...",
        )

    progress.empty()

    if failed_uploads:
        st.toast(
            f"Skipped {len(failed_uploads)} unsupported or invalid file(s). "
            "See the details below.",
            icon="⚠️",
        )

    if not digit_preparations:
        st.session_state.pop("digit_batch_results", None)
        if failed_uploads:
            with st.expander(f"Skipped {len(failed_uploads)} file(s)"):
                for name, reason in failed_uploads:
                    st.warning(f"{name}: {reason}")
        st.error("No valid images could be processed.")
        st.stop()

    input_batch = np.stack(
        [preparation.normalized for preparation in digit_preparations]
    ).reshape(len(digit_preparations), -1)
    batch_probabilities = model.predict_proba(
        input_batch,
        batch_size=min(len(input_batch), 32),
    )
    predicted_digits = np.argmax(batch_probabilities, axis=1)
    confidences = batch_probabilities[
        np.arange(len(batch_probabilities)),
        predicted_digits,
    ]

    result_rows: list[dict[str, int | float | str]] = []
    for filename, preparation, probabilities, predicted_digit, confidence in zip(
        file_names,
        digit_preparations,
        batch_probabilities,
        predicted_digits,
        confidences,
        strict=True,
    ):
        top_indices = np.argsort(probabilities)[::-1][:3]
        result_rows.append(
            {
                "File": filename,
                "Predicted digit": int(predicted_digit),
                "Top probability": float(confidence),
                "2nd choice": int(top_indices[1]),
                "2nd probability": float(probabilities[top_indices[1]]),
                "3rd choice": int(top_indices[2]),
                "3rd probability": float(probabilities[top_indices[2]]),
                "Sharpness score": round(preparation.sharpness_score, 3),
                "Image quality": (
                    "Soft (sharpened)"
                    if preparation.enhancement_applied
                    else "Soft"
                    if preparation.is_blurry
                    else "Sharp"
                ),
            }
        )

    st.session_state["digit_batch_results"] = {
        "digit_preparations": digit_preparations,
        "input_batch": input_batch,
        "file_names": file_names,
        "batch_probabilities": batch_probabilities,
        "predicted_digits": predicted_digits,
        "confidences": confidences,
        "result_rows": result_rows,
        "failed_uploads": failed_uploads,
        "auto_sharpen": auto_sharpen,
    }
    st.session_state["prediction_image_index"] = 0

batch_result = st.session_state.get("digit_batch_results")
if batch_result is not None:
    digit_preparations = batch_result["digit_preparations"]
    input_batch = batch_result["input_batch"]
    file_names = batch_result["file_names"]
    batch_probabilities = batch_result["batch_probabilities"]
    predicted_digits = batch_result["predicted_digits"]
    confidences = batch_result["confidences"]
    result_rows = batch_result["result_rows"]

    failed_uploads = batch_result["failed_uploads"]
    if failed_uploads:
        with st.expander(f"Skipped {len(failed_uploads)} file(s)"):
            for name, reason in failed_uploads:
                st.warning(f"{name}: {reason}")

    st.subheader("Batch results")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Images predicted", len(result_rows))
    metric_columns[1].metric(
        "Average top probability",
        f"{float(np.mean(confidences)):.1%}",
    )
    metric_columns[2].metric(
        "Soft-edge images",
        f"{sum(prep.is_blurry for prep in digit_preparations)} / "
        f"{len(digit_preparations)}",
    )
    metric_columns[3].metric(
        "Different predicted digits",
        len(np.unique(predicted_digits)),
    )

    st.dataframe(result_rows, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=list(result_rows[0]))
    csv_writer.writeheader()
    csv_writer.writerows(result_rows)
    st.download_button(
        "Download predictions as CSV",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="digit-predictions.csv",
        mime="text/csv",
        type="primary",
    )

    probability_figure = plot_probability_cloud_3d(
        batch_probabilities,
        file_names,
    )
    st.plotly_chart(
        probability_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption(
        "**What:** each point is one image's probability for one digit; higher "
        "points indicate stronger classes. **Ask:** are the top choices clear "
        "or are probabilities spread across several digits?"
    )

    quality_figure = go.Figure(
        go.Scatter3d(
            x=confidences,
            y=[
                preparation.sharpness_score
                for preparation in digit_preparations
            ],
            z=predicted_digits,
            mode="markers",
            text=file_names,
            customdata=[
                [
                    float(confidence),
                    preparation.sharpness_score,
                    "Yes" if preparation.is_blurry else "No",
                ]
                for confidence, preparation in zip(
                    confidences,
                    digit_preparations,
                    strict=True,
                )
            ],
            marker=dict(
                size=8,
                color=[
                    1 if preparation.is_blurry else 0
                    for preparation in digit_preparations
                ],
                colorscale=[
                    [0.0, "#5F8F7A"],
                    [0.49, "#5F8F7A"],
                    [0.5, "#B7654F"],
                    [1.0, "#B7654F"],
                ],
                cmin=0,
                cmax=1,
                showscale=False,
                line=dict(color="white", width=1),
            ),
            hovertemplate=(
                "%{text}<br>Predicted digit: %{z}"
                "<br>Top probability: %{customdata[0]:.1%}"
                "<br>Sharpness: %{customdata[1]:.3f}"
                "<br>Soft edges detected: %{customdata[2]}"
                "<extra></extra>"
            ),
            name="Uploaded images",
        )
    )
    quality_figure.update_layout(
        title="3D batch quality map",
        template="plotly_dark",
        scene=dict(
            xaxis_title="Top model probability",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            yaxis_title="Normalized edge sharpness",
            zaxis_title="Predicted digit",
            zaxis=dict(
                tickmode="array",
                tickvals=list(range(model.output_dim)),
            ),
            bgcolor="#0B1017",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        height=520,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(
        quality_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption(
        f"Green markers are above the soft-edge threshold "
        f"({BLUR_SCORE_THRESHOLD:.2f}); red markers were flagged for review."
    )
    st.caption(
        "**What:** each point is an image positioned by model probability, "
        "edge sharpness, and predicted digit. **Ask:** do low-probability "
        "predictions cluster among the soft-edge images?"
    )

    review_threshold = st.slider(
        "Flag images below this top-probability score",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help=(
            "This is a review heuristic, not a calibrated confidence threshold."
        ),
    )
    review_rows = [
        {
            "File": row["File"],
            "Predicted digit": row["Predicted digit"],
            "Top probability": row["Top probability"],
        }
        for row in result_rows
        if row["Top probability"] < review_threshold
    ]
    if review_rows:
        st.warning(
            f"{len(review_rows)} image(s) are below the selected score "
            "and may need manual review."
        )
        st.dataframe(review_rows, use_container_width=True, hide_index=True)
    else:
        st.success("No predictions fall below the selected review score.")

    st.subheader("Inspect an image")
    selected_index = st.selectbox(
        "Choose an image to explore",
        options=list(range(len(file_names))),
        format_func=lambda index: file_names[index],
        key="prediction_image_index",
    )
    selected_logits, selected_cache = model.forward(
        input_batch[selected_index : selected_index + 1]
    )
    selected_probabilities = softmax(selected_logits)[0]
    selected_digit = int(np.argmax(selected_probabilities))
    selected_preparation = digit_preparations[selected_index]

    detail_columns = st.columns(3)
    detail_columns[0].metric("Predicted digit", selected_digit)
    detail_columns[1].metric(
        "Top probability",
        f"{float(selected_probabilities[selected_digit]):.1%}",
    )
    detail_columns[2].metric(
        "Edge sharpness",
        f"{selected_preparation.sharpness_score:.3f}",
        help=(
            "Normalized Laplacian-variance heuristic; lower values suggest "
            "softer edges."
        ),
    )

    if selected_preparation.is_blurry:
        if selected_preparation.enhancement_applied:
            st.info(
                "Soft edges were detected. A gentle unsharp mask was applied "
                "to this image before prediction; it cannot recover detail "
                "that was not captured."
            )
        else:
            st.warning(
                "Soft edges were detected, but auto-sharpening is disabled "
                "or set to zero."
            )
        before_column, after_column = st.columns(2)
        with before_column:
                st.plotly_chart(
                    plot_pixel_surface_3d(
                        np.asarray(
                            selected_preparation.original_crop_preview,
                            dtype=np.float32,
                        ) / 255.0,
                        title="Original crop · pixel-intensity terrain",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
            )
                st.caption("Original crop as a 3D pixel surface")
        with after_column:
                st.plotly_chart(
                    plot_pixel_surface_3d(
                        np.asarray(
                            selected_preparation.processed_crop_preview,
                            dtype=np.float32,
                        ) / 255.0,
                        title="Processed crop · pixel-intensity terrain",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
            )
                st.caption(
                    "Sharpened crop as a 3D pixel surface"
                    if selected_preparation.enhancement_applied
                    else "Crop used for prediction as a 3D pixel surface"
                )
    else:
        st.success("The digit edges look sharp; no enhancement was applied.")

    st.subheader("Inspect the digit as a 3D pixel surface")
    st.caption(
        "**What:** pixel brightness becomes height across the 28 × 28 input. "
        "**Ask:** which strokes create the strongest peaks?"
    )
    st.plotly_chart(
        plot_pixel_surface_3d(
            np.asarray(selected_preparation.canvas, dtype=np.float32) / 255.0
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.subheader("Animate this image's 3D forward pass")
    st.caption(
        "**What:** sampled neuron colors show this image's real activation "
        "values at each layer. **Ask:** how does the signal reach the output?"
        " Rotate, press Play, or move the stage slider."
    )
    st.plotly_chart(
        plot_activation_flow_animation_3d(
            selected_cache.activations,
            selected_probabilities,
            max_units=sidebar_settings.neurons_per_layer,
            frame_duration_ms=sidebar_settings.animation_duration_ms,
        ),
        use_container_width=True,
    )

st.caption(
    "Predictions depend on image quality and similarity to MNIST handwriting. "
    "The displayed probabilities are model outputs, "
    "not calibrated confidence estimates."
)