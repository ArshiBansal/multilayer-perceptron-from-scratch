"""Inference performance measurements for the NumPy MLP."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from src.model import NumpyMLP


def benchmark_inference(
    model: NumpyMLP,
    inputs: np.ndarray,
    batch_sizes: Sequence[int | None],
    *,
    repetitions: int = 3,
) -> list[dict[str, int | float | str]]:
    """Measure model inference at selected batch sizes using median timings.

    A short unrecorded warm-up is run for every batch size. Each timing then
    measures predictions over the complete input matrix.
    """
    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must be a non-empty 2D array.")
    if not batch_sizes:
        raise ValueError("Select at least one batch size.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    valid_batch_sizes: list[int | None] = []
    for batch_size in batch_sizes:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch sizes must be positive or None.")
        if batch_size not in valid_batch_sizes:
            valid_batch_sizes.append(batch_size)

    sample_count = len(inputs)
    results: list[dict[str, int | float | str]] = []
    for batch_size in valid_batch_sizes:
        model.predict_proba(inputs[: min(sample_count, batch_size or sample_count)],
                            batch_size=batch_size)
        durations = []
        for _ in range(repetitions):
            start = time.perf_counter()
            probabilities = model.predict_proba(inputs, batch_size=batch_size)
            durations.append(time.perf_counter() - start)

        if not np.isfinite(probabilities).all():
            raise RuntimeError("Inference returned non-finite probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Inference probabilities do not sum to one.")

        median_seconds = float(np.median(durations))
        mean_seconds = float(np.mean(durations))
        results.append(
            {
                "Batch size": (
                    "Full batch" if batch_size is None else str(batch_size)
                ),
                "Median time (ms)": median_seconds * 1000,
                "Mean time (ms)": mean_seconds * 1000,
                "Latency (ms / image)": median_seconds * 1000 / sample_count,
                "Throughput (images / s)": sample_count / median_seconds,
                "Batches per run": (
                    1
                    if batch_size is None
                    else (sample_count + batch_size - 1) // batch_size
                ),
            }
        )

    return results
