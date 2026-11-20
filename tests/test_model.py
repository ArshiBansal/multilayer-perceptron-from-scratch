"""Smoke-test a saved MNIST MLP artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from mnist_mlp.model import NumpyMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a saved MNIST MLP artifact and run a prediction smoke test."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("models/mnist_mlp.npz"),
        help="model artifact path (default: models/mnist_mlp.npz)",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="also evaluate once on the official MNIST test split",
    )
    return parser


def _resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run_smoke_test(model: NumpyMLP, *, seed: int = 42) -> None:
    """Check that the model produces valid predictions for normalized inputs."""
    rng = np.random.default_rng(seed)
    X = rng.random((8, model.input_dim), dtype=np.float32)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    labels = np.arange(len(X), dtype=np.intp) % model.output_dim
    loss = model.loss(X, labels)

    if probabilities.shape != (len(X), model.output_dim):
        raise AssertionError(
            f"Unexpected probability shape: {probabilities.shape}."
        )
    if predictions.shape != (len(X),):
        raise AssertionError(f"Unexpected prediction shape: {predictions.shape}.")
    if not np.isfinite(probabilities).all():
        raise AssertionError("Predicted probabilities contain non-finite values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Predicted probabilities do not sum to 1.")
    if np.any(predictions < 0) or np.any(predictions >= model.output_dim):
        raise AssertionError("Predictions contain an invalid class index.")
    if not np.isfinite(loss):
        raise AssertionError("Loss is non-finite.")

    print(f"Smoke test passed for {len(X)} synthetic inputs.")
    print(f"Example prediction: {predictions[0]}")
    print(f"Example probabilities: {np.round(probabilities[0], 3)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_path = _resolve_path(args.artifact)

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    model = NumpyMLP.load(artifact_path)
    if model.input_dim != 784 or model.output_dim != 10:
        raise ValueError(
            "Expected an MNIST model with 784 inputs and 10 output classes; "
            f"got {model.input_dim} inputs and {model.output_dim} outputs."
        )

    print(f"Loaded model artifact: {artifact_path}")
    print(
        f"Architecture: 784 -> {' -> '.join(map(str, model.hidden_dims))} -> 10"
    )
    run_smoke_test(model)

    if args.evaluate_test:
        from mnist_mlp.data import load_mnist
        from mnist_mlp.evaluate import evaluate

        (_, _), (X_test, y_test) = load_mnist()
        report = evaluate(model, X_test, y_test)
        print(f"Test loss: {report.loss:.4f}")
        print(f"Test accuracy: {report.accuracy:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())