"""Tests for digit-image validation, blur detection, and enhancement."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFilter, UnidentifiedImageError

from src.image_processing import (
    calculate_sharpness,
    has_supported_image_extension,
    prepare_digit,
)


def _digit_image() -> Image.Image:
    image = Image.new("L", (128, 128), color=0)
    draw = ImageDraw.Draw(image)
    draw.line(
        [(45, 98), (45, 30), (86, 30), (86, 62), (45, 62)],
        fill=255,
        width=12,
        joint="curve",
    )
    return image


def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "filename",
    [
        "digit.png",
        "digit.JPG",
        "digit.jfif",
        "digit.bmp",
        "digit.dib",
        "digit.gif",
        "digit.tiff",
        "digit.webp",
    ],
)
def test_supported_image_extensions_are_case_insensitive(filename: str) -> None:
    assert has_supported_image_extension(filename)


@pytest.mark.parametrize("filename", ["notes.pdf", "digit.svg", "unknown"])
def test_unsupported_image_extensions_are_rejected(filename: str) -> None:
    assert not has_supported_image_extension(filename)


def test_invalid_image_contents_are_rejected() -> None:
    with pytest.raises(UnidentifiedImageError):
        prepare_digit(b"not an image", invert=False)


def test_blurry_digit_is_detected_and_auto_sharpened() -> None:
    sharp_image = _digit_image()
    blurry_image = sharp_image.filter(ImageFilter.GaussianBlur(radius=4))

    assert calculate_sharpness(blurry_image) < calculate_sharpness(sharp_image)

    result = prepare_digit(
        _image_bytes(blurry_image),
        invert=False,
        auto_sharpen=True,
        sharpening_amount=125,
    )

    assert result.is_blurry
    assert result.enhancement_applied
    assert result.canvas.size == (28, 28)
    assert result.normalized.shape == (28, 28)
    assert np.isfinite(result.normalized).all()
    assert calculate_sharpness(result.processed_crop_preview) > calculate_sharpness(
        result.original_crop_preview
    )
    assert ImageChops.difference(
        result.original_crop_preview,
        result.processed_crop_preview,
    ).getbbox()


def test_blur_detection_does_not_apply_filter_when_disabled() -> None:
    blurry_image = _digit_image().filter(ImageFilter.GaussianBlur(radius=4))

    result = prepare_digit(
        _image_bytes(blurry_image),
        invert=False,
        auto_sharpen=False,
    )

    assert result.is_blurry
    assert not result.enhancement_applied
