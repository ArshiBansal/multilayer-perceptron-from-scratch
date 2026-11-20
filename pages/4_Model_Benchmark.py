"""Benchmark inference speed and model footprint."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.benchmark import benchmark_inference
from src.model import NumpyMLP
from src.sidebar import render_sidebar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.npz"
BATCH_SIZE_OPTIONS = (1, 16, 64, 256, 1024)
MAX_BENCHMARK_SAMPLES = 2048


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path: str, artifact_version: tuple[int, int]) -> NumpyMLP:
    """Cache the saved model for this Streamlit process."""
    del artifact_version
    return NumpyMLP.load(model_path)


def get_model_operations(model: NumpyMLP) -> int:
    """Estimate multiply-accumulate operations for one forward pass."""
    return sum(
        model.params[f"W{layer_index}"].size
        for layer_index in range(1, model.layer_count + 1)
    )


st.set_page_config(
    page_title="Model Benchmark",
    page_icon="⏱️",
    layout="wide",
)
sidebar_settings = render_sidebar()
if sidebar_settings.model_path is not None:
    MODEL_PATH = sidebar_settings.model_path

st.title("Model Benchmark")
st.markdown(
    "Measure **inference latency and throughput** across batch sizes, then "
    "inspect the saved model's parameter and artifact footprint."
)
st.info(
    "Measurements run on the machine serving this Streamlit app. Inputs are "
    "deterministic synthetic pixel values, not MNIST examples; results describe "
    "this runtime and are not a hardware-independent performance guarantee."
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
    st.stop()

parameter_count = sum(parameter.size for parameter in model.params.values())
parameter_bytes = sum(parameter.nbytes for parameter in model.params.values())
artifact_bytes = MODEL_PATH.stat().st_size
operations_per_image = get_model_operations(model)

layer_dimensions = (model.input_dim, *model.hidden_dims, model.output_dim)
layer_labels = [
    f"{layer_dimensions[index]:,} → {layer_dimensions[index + 1]:,}"
    for index in range(model.layer_count)
]
layer_macs = [
    model.params[f"W{index}"].size
    for index in range(1, model.layer_count + 1)
]
layer_parameters = [
    model.params[f"W{index}"].size + model.params[f"b{index}"].size
    for index in range(1, model.layer_count + 1)
]
layer_weight_bytes = [
    model.params[f"W{index}"].nbytes + model.params[f"b{index}"].nbytes
    for index in range(1, model.layer_count + 1)
]

profile_figure = go.Figure(
    go.Bar(
        x=layer_macs,
        y=layer_labels,
        orientation="h",
        text=[f"{count / 1_000_000:.2f} M" for count in layer_macs],
        textposition="outside",
        customdata=list(zip(layer_parameters, layer_weight_bytes, strict=True)),
        marker=dict(
            color=[
                ("#527F94", "#5F8F7A", "#6675A8")[index % 3]
                for index in range(len(layer_labels))
            ],
            line=dict(color="rgba(255,255,255,.35)", width=1),
        ),
        hovertemplate=(
            "Layer: %{y}<br>MACs / image: %{x:,}"
            "<br>Parameters: %{customdata[0]:,}"
            "<br>Weight memory: %{customdata[1]:,} bytes<extra></extra>"
        ),
    )
)
profile_figure.update_layout(
    title="Compute signature · work by layer",
    template="plotly_dark",
    xaxis_title="Multiply-accumulates per image",
    yaxis_title="Layer dimensions",
    yaxis=dict(autorange="reversed"),
    showlegend=False,
    height=max(280, 68 * len(layer_labels)),
    margin=dict(l=20, r=50, t=55, b=35),
)

st.subheader("Efficiency fingerprint")
st.caption(
    "Each bar shows the compute performed by one layer for a single image. "
    "Hover for parameter count and weight memory."
)
with st.container(border=True):
    st.plotly_chart(
        profile_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption(
        f"Total: {operations_per_image:,} MACs / image · "
        f"{parameter_count:,} parameters · "
        f"{parameter_bytes / 1024:.1f} KiB in parameter arrays · "
        f"{artifact_bytes / 1024:.1f} KiB on disk."
    )

with st.form("model_benchmark_form"):
    st.subheader("Run an inference benchmark")
    settings_columns = st.columns(3)
    with settings_columns[0]:
        sample_count = st.select_slider(
            "Synthetic input examples",
            options=(128, 256, 512, 1024, MAX_BENCHMARK_SAMPLES),
            value=512,
        )
    with settings_columns[1]:
        selected_batch_sizes = st.multiselect(
            "Batch sizes",
            options=BATCH_SIZE_OPTIONS,
            default=(16, 64, 256, 1024),
            help="Sizes larger than the input set are ignored for chunking.",
        )
    with settings_columns[2]:
        repetitions = st.slider(
            "Measured repetitions",
            min_value=1,
            max_value=7,
            value=3,
            help="Each selected batch size gets one warm-up and these timed runs.",
        )
    run_benchmark = st.form_submit_button(
        "Run benchmark",
        type="primary",
    )

if run_benchmark:
    if not selected_batch_sizes:
        st.session_state.pop("model_benchmark_results", None)
        st.warning("Select at least one batch size.")
    else:
        benchmark_sizes: list[int | None] = list(
            dict.fromkeys(
                None if size >= sample_count else size
                for size in selected_batch_sizes
            )
        )
        if None not in benchmark_sizes:
            benchmark_sizes.append(None)
        rng = np.random.default_rng(42)
        synthetic_inputs = rng.random(
            (sample_count, model.input_dim),
            dtype=np.float32,
        )

        with st.spinner(
            f"Running {len(benchmark_sizes)} batch-size tests "
            f"({repetitions} repetitions each)..."
        ):
            try:
                benchmark_rows = benchmark_inference(
                    model,
                    synthetic_inputs,
                    benchmark_sizes,
                    repetitions=repetitions,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                st.session_state.pop("model_benchmark_results", None)
                st.error(f"Benchmark could not be completed: {exc}")
            else:
                st.session_state["model_benchmark_results"] = {
                    "rows": benchmark_rows,
                    "sample_count": sample_count,
                    "repetitions": repetitions,
                    "architecture": (
                        model.input_dim,
                        *model.hidden_dims,
                        model.output_dim,
                    ),
                    "dtype": str(model.dtype),
                }

benchmark_result = st.session_state.get("model_benchmark_results")
if benchmark_result is not None:
    result_rows = benchmark_result["rows"]
    st.subheader("Benchmark results")
    st.caption(
        f"{benchmark_result['sample_count']:,} deterministic synthetic inputs · "
        f"{benchmark_result['repetitions']} measured repetitions · "
        f"dtype {benchmark_result['dtype']} · "
        f"architecture {' → '.join(map(str, benchmark_result['architecture']))}"
    )

    batch_labels = [str(row["Batch size"]) for row in result_rows]

    sample_count = benchmark_result["sample_count"]
    effective_batch_sizes = [
        sample_count if label == "Full batch" else int(label)
        for label in batch_labels
    ]
    tradeoff_figure = go.Figure(
        go.Scatter3d(
            x=effective_batch_sizes,
            y=[row["Median time (ms)"] for row in result_rows],
            z=[row["Throughput (images / s)"] for row in result_rows],
            mode="markers+text",
            text=batch_labels,
            textposition="top center",
            customdata=[
                [row["Latency (ms / image)"], row["Batches per run"]]
                for row in result_rows
            ],
            marker=dict(
                size=8,
                color=[row["Throughput (images / s)"] for row in result_rows],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Images / s"),
                line=dict(color="white", width=1),
            ),
            hovertemplate=(
                "Batch size: %{text}<br>Median run time: %{y:.3f} ms"
                "<br>Throughput: %{z:.1f} images / s"
                "<br>Latency: %{customdata[0]:.4f} ms / image"
                "<br>Batches: %{customdata[1]}<extra></extra>"
            ),
            name="Batch sizes",
        )
    )
    tradeoff_figure.update_layout(
        title="3D inference performance landscape",
        template="plotly_dark",
        scene=dict(
            xaxis_title="Batch size",
            yaxis_title="Median time per input set (ms)",
            zaxis_title="Throughput (images / s)",
            bgcolor="#0B1017",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        height=560,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    st.plotly_chart(
        tradeoff_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption(
        "**What:** batch size, total run time, and throughput in one view; "
        "hover for per-image latency. **Ask:** where is the best speed "
        "trade-off for this machine?"
    )

    st.dataframe(result_rows, use_container_width=True, hide_index=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(result_rows[0]))
    writer.writeheader()
    writer.writerows(result_rows)
    st.download_button(
        "Download benchmark results",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="mlp-inference-benchmark.csv",
        mime="text/csv",
    )

with st.expander("How to interpret the results"):
    st.markdown(
        """
        - **Latency per image** divides the median end-to-end batch time by the
          number of examples; it is not the time for an isolated request.
        - **Throughput** is the number of examples processed per second across
          each complete test run.
        - **Full batch** sends all selected examples through the model at once;
          smaller batch sizes split them into multiple forward passes.
        - Results depend on the serving machine, NumPy/BLAS implementation,
          concurrent load, and selected input count. Rerun under the same
          conditions when comparing model changes.
        """
    )