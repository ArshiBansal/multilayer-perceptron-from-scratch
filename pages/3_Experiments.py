"""Streamlit page for configuring and comparing MNIST experiments."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments.csv"

REQUIRED_COLUMNS = {"name", "group", "val_accuracy", "val_loss"}


def parse_experiment_results(csv_text: str) -> list[dict[str, Any]]:
    """Parse and validate the experiment-results CSV used by notebook 02."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("The CSV is empty or has no header row.")

    missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}.")

    rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            name = (row.get("name") or "").strip()
            group = (row.get("group") or "").strip()
            accuracy = float(row["val_accuracy"])
            loss = float(row["val_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid values on CSV line {line_number}.") from exc

        if not name or not group:
            raise ValueError(
                f"CSV line {line_number} must include a name and group."
            )
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(
                f"Validation accuracy on line {line_number} must be in [0, 1]."
            )
        if not math.isfinite(loss) or loss < 0:
            raise ValueError(
                f"Validation loss on line {line_number} must be finite and non-negative."
            )

        experiment: dict[str, Any] = {
            "name": name,
            "group": group,
            "val_accuracy": accuracy,
            "val_loss": loss,
        }

        for column in ("hidden_dims", "activation", "initialization"):
            if row.get(column):
                experiment[column] = row[column].strip()

        for column in ("parameters", "batch_size"):
            if row.get(column):
                try:
                    experiment[column] = int(row[column])
                except ValueError as exc:
                    raise ValueError(
                        f"{column} on CSV line {line_number} must be an integer."
                    ) from exc

        if row.get("seconds"):
            try:
                seconds = float(row["seconds"])
            except ValueError as exc:
                raise ValueError(
                    f"seconds on CSV line {line_number} must be numeric."
                ) from exc
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    f"seconds on CSV line {line_number} must be finite and non-negative."
                )
            experiment["seconds"] = seconds

        rows.append(experiment)

    if not rows:
        raise ValueError("The CSV contains a header but no experiment rows.")

    return rows


def make_grouped_bar(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    y_axis_title: str,
) -> go.Figure:
    groups = list(dict.fromkeys(row["group"] for row in rows))
    figure = go.Figure()

    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        figure.add_trace(
            go.Bar(
                name=group,
                x=[row["name"] for row in group_rows],
                y=[row[metric] for row in group_rows],
                hovertemplate=(
                    "Experiment: %{x}<br>"
                    f"{y_axis_title}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Experiment",
        yaxis_title=y_axis_title,
        barmode="group",
        xaxis_tickangle=-25,
    )
    if metric == "val_accuracy":
        figure.update_yaxes(tickformat=".0%", range=[0, 1])

    return figure


st.set_page_config(
    page_title="Experiments",
    page_icon="🧪",
    layout="wide",
)

st.title("Experiment Lab")
st.write(
    "Configure a reproducible training run, then compare validation results "
    "exported from `02_Training_Experiments.ipynb`."
)

st.subheader("Configure a training run")

with st.expander("Training settings", expanded=True):
    architecture_text = st.text_input(
        "Hidden-layer widths",
        value="128",
        help="Enter one or more comma-separated widths, for example 256,128.",
    )

    columns = st.columns(3)
    with columns[0]:
        activation = st.selectbox("Activation", ("relu", "sigmoid"))
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=10)
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=10_000,
            value=128,
        )
    with columns[1]:
        initialization = st.selectbox("Initialization", ("he", "xavier"))
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.000001,
            max_value=10.0,
            value=0.1,
            format="%.6f",
        )
        momentum = st.number_input(
            "Momentum",
            min_value=0.0,
            max_value=0.99,
            value=0.0,
            step=0.05,
        )
    with columns[2]:
        l2_strength = st.number_input(
            "L2 strength",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            format="%.6f",
        )
        validation_size = st.number_input(
            "Validation examples",
            min_value=10,
            max_value=59_990,
            value=5_000,
            step=500,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

    try:
        hidden_dims = tuple(
            int(part.strip()) for part in architecture_text.split(",")
        )
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError
        architecture_error = None
    except ValueError:
        architecture_error = (
            "Enter positive integer widths separated by commas, "
            "for example `128` or `256,128`."
        )

if architecture_error:
    st.error(architecture_error)
else:
    command = (
        "python scripts/train.py "
        f"--hidden-dims {','.join(map(str, hidden_dims))} "
        f"--activation {activation} "
        f"--initialization {initialization} "
        f"--epochs {int(epochs)} "
        f"--batch-size {int(batch_size)} "
        f"--learning-rate {learning_rate:.8g} "
        f"--momentum {momentum:.8g} "
        f"--l2-strength {l2_strength:.8g} "
        f"--validation-size {int(validation_size)} "
        f"--seed {int(seed)}"
    )
    st.markdown("Run this command from the project root:")
    st.code(command, language="powershell")
    st.caption(
        "The command trains and saves a model artifact. It does not evaluate "
        "on the official test set unless you explicitly add `--evaluate-test`."
    )

st.divider()
st.subheader("Compare experiment results")

uploaded_csv = st.file_uploader(
    "Upload an experiment-results CSV",
    type=("csv",),
    help=(
        "Expected columns include name, group, val_accuracy, and val_loss. "
        "Notebook 02 also records architecture and runtime details."
    ),
)

try:
    if uploaded_csv is not None:
        csv_text = uploaded_csv.getvalue().decode("utf-8-sig")
        experiment_rows = parse_experiment_results(csv_text)
        result_source = uploaded_csv.name
    elif RESULTS_PATH.is_file():
        experiment_rows = parse_experiment_results(
            RESULTS_PATH.read_text(encoding="utf-8-sig")
        )
        result_source = str(RESULTS_PATH.relative_to(PROJECT_ROOT))
    else:
        experiment_rows = []
        result_source = ""
except (UnicodeDecodeError, OSError, csv.Error, ValueError) as exc:
    st.error(f"Could not load experiment results: {exc}")
    experiment_rows = []
    result_source = ""

if experiment_rows:
    st.caption(f"Results source: `{result_source}`")

    best = max(experiment_rows, key=lambda row: row["val_accuracy"])
    metric_columns = st.columns(3)
    metric_columns[0].metric("Experiments", len(experiment_rows))
    metric_columns[1].metric(
        "Best validation accuracy",
        f"{best['val_accuracy']:.2%}",
    )
    metric_columns[2].metric("Best run", best["name"])

    st.dataframe(experiment_rows, use_container_width=True, hide_index=True)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.plotly_chart(
            make_grouped_bar(
                experiment_rows,
                metric="val_accuracy",
                title="Validation accuracy",
                y_axis_title="Accuracy",
            ),
            use_container_width=True,
        )
    with chart_columns[1]:
        st.plotly_chart(
            make_grouped_bar(
                experiment_rows,
                metric="val_loss",
                title="Validation loss",
                y_axis_title="Cross-entropy loss",
            ),
            use_container_width=True,
        )
else:
    st.info(
        "No experiment results are available yet. Run the experiments in "
        "`02_Training_Experiments.ipynb`, then export its `results` list."
    )

st.markdown("**Notebook 02 export cell**")
st.code(
    '''import csv

results_path = PROJECT_ROOT / "reports" / "experiments.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

with results_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved experiment results to: {results_path}")''',
    language="python",
)