"""Tests for bounded, reproducible model-inference benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import benchmark_inference
from src.model import NumpyMLP


def _small_model() -> NumpyMLP:
    return NumpyMLP(
        input_dim=4,
        hidden_dims=(3,),
        output_dim=2,
        seed=7,
    )


def test_benchmark_reports_finite_latency_and_throughput() -> None:
    model = _small_model()
    inputs = np.random.default_rng(42).random((8, 4), dtype=np.float32)

    results = benchmark_inference(
        model,
        inputs,
        [1, 4, None],
        repetitions=2,
    )

    assert [row["Batch size"] for row in results] == ["1", "4", "Full batch"]
    assert results[0]["Batches per run"] == 8
    assert results[1]["Batches per run"] == 2
    assert results[2]["Batches per run"] == 1
    for row in results:
        assert np.isfinite(row["Median time (ms)"])
        assert row["Median time (ms)"] > 0
        assert np.isfinite(row["Latency (ms / image)"])
        assert row["Latency (ms / image)"] > 0
        assert np.isfinite(row["Throughput (images / s)"])
        assert row["Throughput (images / s)"] > 0


@pytest.mark.parametrize(
    ("inputs", "batch_sizes", "repetitions"),
    [
        (np.empty((0, 4), dtype=np.float32), [1], 1),
        (np.zeros((1, 4), dtype=np.float32), [], 1),
        (np.zeros((1, 4), dtype=np.float32), [1], 0),
        (np.zeros((1, 4), dtype=np.float32), [0], 1),
    ],
)
def test_benchmark_rejects_invalid_settings(
    inputs: np.ndarray,
    batch_sizes: list[int],
    repetitions: int,
) -> None:
    with pytest.raises(ValueError):
        benchmark_inference(
            _small_model(),
            inputs,
            batch_sizes,
            repetitions=repetitions,
        )
