"""Image validation and preprocessing for the MNIST digit predictor."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".jfif",
        ".bmp",
        ".dib",
        ".gif",
        ".tif",
        ".tiff",
        ".webp",
    }
)
BLUR_SCORE_THRESHOLD = 0.12
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 10_000
MNIST_CANVAS_SIZE = 28
DIGIT_SIZE = 20


@dataclass(frozen=True)
class DigitPreparation:
    """Prepared model input and bounded previews for image-quality feedback."""

    canvas: Image.Image
    normalized: np.ndarray
    sharpness_score: float
    is_blurry: bool
    enhancement_applied: bool
    original_crop_preview: Image.Image
    processed_crop_preview: Image.Image


def has_supported_image_extension(filename: str) -> bool:
    """Return whether a filename uses one of the supported raster formats."""
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def calculate_sharpness(image: Image.Image) -> float:
    """Estimate edge sharpness with normalized Laplacian variance."""
    sample = image.convert("L").copy()
    sample.thumbnail((128, 128), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.float32)
    if min(pixels.shape) < 3:
        return 0.0

    center = pixels[1:-1, 1:-1]
    laplacian = (
        4 * center
        - pixels[:-2, 1:-1]
        - pixels[2:, 1:-1]
        - pixels[1:-1, :-2]
        - pixels[1:-1, 2:]
    )
    contrast = float(np.var(pixels))
    if contrast <= 1e-6:
        return 0.0
    return float(np.var(laplacian) / contrast)


def _preview(image: Image.Image) -> Image.Image:
    preview = image.copy()
    preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
    return preview


def prepare_digit(
    image_bytes: bytes,
    *,
    invert: bool,
    auto_sharpen: bool = True,
    sharpening_amount: int = 125,
) -> DigitPreparation:
    """Center one digit on an MNIST canvas and optionally sharpen soft edges."""
    if not 0 <= sharpening_amount <= 250:
        raise ValueError("Sharpening amount must be between 0 and 250.")

    with Image.open(BytesIO(image_bytes)) as source:
        if (
            source.width > MAX_IMAGE_SIDE
            or source.height > MAX_IMAGE_SIDE
            or source.width * source.height > MAX_IMAGE_PIXELS
        ):
            raise ValueError("Image dimensions are too large.")
        oriented = ImageOps.exif_transpose(source)
        grayscale = oriented.convert("L")

    if invert:
        grayscale = ImageOps.invert(grayscale)

    foreground = grayscale.point(lambda pixel: 255 if pixel > 30 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise ValueError("No visible digit found. Try a clearer image.")

    original_crop = grayscale.crop(bounds)
    width, height = original_crop.size
    if width == 0 or height == 0:
        raise ValueError("The uploaded image does not contain a usable digit.")

    sharpness = calculate_sharpness(original_crop)
    is_blurry = sharpness < BLUR_SCORE_THRESHOLD
    enhancement_applied = is_blurry and auto_sharpen and sharpening_amount > 0
    processed_crop = (
        original_crop.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=sharpening_amount,
                threshold=3,
            )
        )
        if enhancement_applied
        else original_crop
    )

    scale = DIGIT_SIZE / max(width, height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = processed_crop.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE), color=0)
    offset = (
        (MNIST_CANVAS_SIZE - resized.width) // 2,
        (MNIST_CANVAS_SIZE - resized.height) // 2,
    )
    canvas.paste(resized, offset)

    normalized = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    return DigitPreparation(
        canvas=canvas,
        normalized=normalized,
        sharpness_score=sharpness,
        is_blurry=is_blurry,
        enhancement_applied=enhancement_applied,
        original_crop_preview=_preview(original_crop),
        processed_crop_preview=_preview(processed_crop),
    )