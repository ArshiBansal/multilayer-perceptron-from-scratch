"""Download MNIST through the Keras dataset loader and validate the cached data.

Install the project first, for example:
    python -m pip install -r requirements-train.txt

Then run:
    python scripts/download_data.py

To use a custom Keras cache root:
    python scripts/download_data.py --keras-home PATH
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from mnist_mlp.data import load_mnist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load MNIST using Keras. The dataset is downloaded if needed "
            "and stored in the Keras cache."
        )
    )
    parser.add_argument(
        "--keras-home",
        type=Path,
        help=(
            "Keras cache root. Relative paths are resolved from the project "
            "root. MNIST is stored in this directory's datasets subfolder."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.keras_home is not None:
        keras_home = args.keras_home
        if not keras_home.is_absolute():
            keras_home = PROJECT_ROOT / keras_home
        keras_home = keras_home.expanduser().resolve()

        # Set before load_mnist imports TensorFlow/Keras.
        os.environ["KERAS_HOME"] = str(keras_home)
    else:
        configured_home = os.environ.get("KERAS_HOME")
        keras_home = (
            Path(configured_home).expanduser()
            if configured_home
            else Path.home() / ".keras"
        ).resolve()

    print("Loading MNIST with the Keras dataset loader...")
    print(f"Keras cache root: {keras_home}")
    print("A download will occur only if the dataset is not already cached.")

    (X_train, y_train), (X_test, y_test) = load_mnist()

    print("MNIST is ready.")
    print(f"Training images: {X_train.shape}, dtype={X_train.dtype}")
    print(f"Training labels: {y_train.shape}, dtype={y_train.dtype}")
    print(f"Test images:     {X_test.shape}, dtype={X_test.dtype}")
    print(f"Test labels:     {y_test.shape}, dtype={y_test.dtype}")
    print(f"Pixel range:     {X_train.min()}–{X_train.max()}")
    print("The dataset remains in the Keras cache; no files were copied into the project.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())