"""Configure reproducible CLI training runs and compare their results."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import streamlit as st

from src.experiment_visualization import (
    get_parameter_count,
    make_experiment_tradeoff_3d,
)
from src.sidebar import render_sidebar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments.csv"
REQUIRED_COLUMNS = {"name", "group", "val_accuracy", "val_loss"}
RESULT_TEMPLATE = (
    "name,group,val_accuracy,val_loss,hidden_dims,activation,"
    "initialization,parameters,batch_size,seconds\n"
    "relu-baseline,activation,0.965,0.12,128,relu,he,101770,128,42.5\n"
    "sigmoid-baseline,activation,0.951,0.17,128,sigmoid,xavier,101770,128,44.1\n"
)


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
                f"Validation loss on line {line_number} must be finite "
                "and non-negative."
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
                    f"seconds on CSV line {line_number} must be finite "
                    "and non-negative."
                )
            experiment["seconds"] = seconds

        rows.append(experiment)

    if not rows:
        raise ValueError("The CSV contains a header but no experiment rows.")
    return rows


st.set_page_config(
    page_title="Training & Experiments",
    page_icon="🧪",
    layout="wide",
)
render_sidebar()

st.title("Training & Experiments")
st.markdown(
    "Build **reproducible training commands**, then compare validation runs "
    "to understand the trade-offs between accuracy, loss, and runtime."
)
st.info(
    "Training runs execute from your project terminal, not inside this page. "
    "This keeps long runs visible and avoids silently replacing the model "
    "currently used by the app."
)

configure_tab, compare_tab = st.tabs(
    ["Configure a training run", "Compare experiments"]
)

with configure_tab:
    st.subheader("Design a reproducible run")
    st.caption(
        "The command uses the project's NumPy trainer and writes a model "
        "artifact. Review it before running from the repository root."
    )

    architecture_text = st.text_input(
        "Hidden-layer widths",
        value="128",
        help="Use comma-separated positive widths, for example 256,128.",
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
            help="Reserved from the 60,000-example training pool.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

    evaluate_test = st.checkbox(
        "Evaluate once on the official test set",
        value=False,
        help=(
            "Keep this unchecked while comparing configurations; use the test "
            "set only after choosing a final model."
        ),
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
        if evaluate_test:
            command += " --evaluate-test"

        st.markdown("#### Generated command")
        st.code(command, language="powershell")
        st.caption(
            "Training saves `models/mnist_mlp.npz`. The app reads that artifact "
            "on a page refresh; the official test split is not used unless "
            "`--evaluate-test` is selected."
        )

with compare_tab:
    st.subheader("Compare validation runs")
    st.caption(
        "Compare runs by their validation metrics. Use a held-out test set only "
        "for final model selection."
    )

    uploaded_csv = st.file_uploader(
        "Upload an experiment-results CSV",
        type=("csv",),
        help=(
            "Required columns: name, group, val_accuracy, and val_loss. "
            "Architecture and runtime columns are optional."
        ),
        key="experiment_results_csv",
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
        groups = list(dict.fromkeys(row["group"] for row in experiment_rows))
        selected_groups = st.multiselect(
            "Experiment groups",
            options=groups,
            default=groups,
        )
        visible_rows = [
            row for row in experiment_rows if row["group"] in selected_groups
        ]

        if not visible_rows:
            st.warning("Select at least one experiment group to compare.")
        else:
            best_accuracy = max(visible_rows, key=lambda row: row["val_accuracy"])
            best_loss = min(visible_rows, key=lambda row: row["val_loss"])
            runtime_rows = [
                row for row in visible_rows if "seconds" in row
            ]

            metric_columns = st.columns(4)
            metric_columns[0].metric("Visible runs", len(visible_rows))
            metric_columns[1].metric(
                "Best validation accuracy",
                f"{best_accuracy['val_accuracy']:.2%}",
            )
            metric_columns[2].metric(
                "Lowest validation loss",
                f"{best_loss['val_loss']:.4f}",
            )
            metric_columns[3].metric(
                "Fastest recorded run",
                (
                    f"{min(row['seconds'] for row in runtime_rows):.1f} s"
                    if runtime_rows
                    else "Not recorded"
                ),
            )

            complexity_rows = [
                row
                for row in visible_rows
                if get_parameter_count(row) is not None
            ]
            if complexity_rows:
                plot_rows = complexity_rows
                depth_metric = "parameters"
                depth_description = "model size"
            else:
                plot_rows = [
                    row for row in visible_rows if "seconds" in row
                ]
                depth_metric = "seconds"
                depth_description = "training time"

            if plot_rows:
                st.plotly_chart(
                    make_experiment_tradeoff_3d(
                        plot_rows,
                        depth_metric=depth_metric,
                    ),
                    width="stretch",
                    config={"displaylogo": False},
                )
                st.caption(
                    f"**What:** {depth_description}, validation loss, and "
                    "accuracy for each run. **Ask:** is the accuracy gain "
                    "worth the added model size or training time?"
                )
            else:
                st.info(
                    "Add `parameters` or `hidden_dims`, or record `seconds`, "
                    "to visualize the validation trade-off in 3D."
                )

            sort_choice = st.selectbox(
                "Sort comparison table by",
                options=(
                    "Validation accuracy (high to low)",
                    "Validation loss (low to high)",
                    "Runtime (fast to slow)",
                ),
            )
            sort_keys = {
                "Validation accuracy (high to low)": ("val_accuracy", True),
                "Validation loss (low to high)": ("val_loss", False),
                "Runtime (fast to slow)": ("seconds", False),
            }
            sort_key, descending = sort_keys[sort_choice]
            sortable_rows = [
                row for row in visible_rows if sort_key in row
            ]
            sorted_rows = sorted(
                sortable_rows or visible_rows,
                key=lambda row: row.get(sort_key, 0),
                reverse=descending,
            )
            st.dataframe(
                sorted_rows,
                use_container_width=True,
                hide_index=True,
            )

            export_fields = list(
                dict.fromkeys(
                    field for row in sorted_rows for field in row.keys()
                )
            )
            export_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(export_buffer, fieldnames=export_fields)
            writer.writeheader()
            writer.writerows(sorted_rows)
            st.download_button(
                "Download visible comparison as CSV",
                data=export_buffer.getvalue().encode("utf-8-sig"),
                file_name="experiment-comparison.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "No experiment results are available yet. Upload a CSV or save "
            "results to `reports/experiments.csv`."
        )
        st.download_button(
            "Download CSV template",
            data=RESULT_TEMPLATE.encode("utf-8"),
            file_name="experiment-results-template.csv",
            mime="text/csv",
        )
