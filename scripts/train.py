"""Train the MNIST NumPy MLP from the command line.

Install the project first, for example:
    python -m pip install -r requirements-train.txt

Then run:
    python scripts/train.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist, split_train_validation  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.model import NumpyMLP  # noqa: E402
from src.train import TrainingConfig, train_model  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _momentum(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)")
    return parsed


def _hidden_dims(value: str) -> tuple[int, ...]:
    try:
        dimensions = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated positive integers, for example 128,64"
        ) from exc

    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be positive integers"
        )
    return dimensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NumPy multilayer perceptron on MNIST."
    )
    parser.add_argument(
        "--hidden-dims",
        type=_hidden_dims,
        default=(128,),
        metavar="N[,N...]",
        help="hidden-layer widths (default: 128; example: 256,128)",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "sigmoid"),
        default="relu",
        help="hidden-layer activation (default: relu)",
    )
    parser.add_argument(
        "--initialization",
        choices=("he", "xavier"),
        default="he",
        help="weight initialization (default: he)",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=10,
        help="number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=128,
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=0.1,
        help="SGD learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=_momentum,
        default=0.0,
        help="SGD momentum in [0, 1) (default: 0)",
    )
    parser.add_argument(
        "--l2-strength",
        type=_nonnegative_float,
        default=0.0,
        help="L2 regularization strength (default: 0)",
    )
    parser.add_argument(
        "--validation-size",
        type=_positive_int,
        default=5_000,
        help="number of training examples reserved for validation (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mnist_mlp.npz"),
        help="model artifact path, relative to project root unless absolute",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the final model once on the official test split",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()

    (X_pool, y_pool), (X_test, y_test) = load_mnist()
    (X_train, y_train), (X_validation, y_validation) = split_train_validation(
        X_pool,
        y_pool,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    print(
        f"Data split: {len(y_train):,} train, "
        f"{len(y_validation):,} validation"
    )
    print(f"Architecture: 784 -> {' -> '.join(map(str, args.hidden_dims))} -> 10")

    model = NumpyMLP(
        input_dim=784,
        hidden_dims=args.hidden_dims,
        output_dim=10,
        activation=args.activation,
        initialization=args.initialization,
        seed=args.seed,
    )
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        l2_strength=args.l2_strength,
        seed=args.seed,
    )

    history = train_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        config=config,
    )
    final_epoch = history.epochs[-1]

    metadata: dict[str, int | float | str | bool] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "final_validation_accuracy": final_epoch.validation_accuracy,
    }

    if args.evaluate_test:
        test_report = evaluate(model, X_test, y_test, batch_size=args.batch_size)
        metadata["test_accuracy"] = test_report.accuracy
        print(
            f"Test loss: {test_report.loss:.4f} | "
            f"test accuracy: {test_report.accuracy:.2%}"
        )

    saved_path = model.save(
        output_path,
        metadata=metadata,
    )
    print(f"Saved model: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())