"""Cover faces with blur, a black rectangle, or pixelation."""

from __future__ import annotations

import cv2
import numpy as np

from noface.errors import NofaceError
from noface.geometry import expand_box
from noface.types import Options

STYLES = ("blur", "black", "pixelate")


def apply_anonymization(
    frame: np.ndarray,
    hide_boxes: list[tuple[float, float, float, float]],
    keep_boxes: list[tuple[float, float, float, float]],
    options: Options,
) -> np.ndarray:
    """Return a copy of ``frame`` with ``hide_boxes`` anonymized.

    Pixels inside ``keep_boxes`` are copied back afterwards, so an expanded
    stranger box that overlaps someone we are keeping does not erase that face.
    """

    if options.style not in STYLES:
        raise NofaceError(f"Unknown style {options.style!r}. Choose from: {', '.join(STYLES)}.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise NofaceError(f"Expected a color frame, got shape {tuple(frame.shape)}.")

    output = frame.copy()
    height, width = frame.shape[:2]
    for box in hide_boxes:
        region = expand_box(box, options.pad, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        view = output[y1:y2, x1:x2]
        if options.style == "black":
            view[:] = 0
        elif options.style == "blur":
            view[:] = _blur(view, options.blur_strength)
        else:
            view[:] = _pixelate(view, options.pixel_size)

    for box in keep_boxes:
        region = expand_box(box, options.restore_pad, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return output


def _blur(region: np.ndarray, strength: int) -> np.ndarray:
    """Downscale and scale back up. Higher ``strength`` removes more detail."""

    height, width = region.shape[:2]
    divisor = max(1, strength)
    small_w = max(1, width // divisor)
    small_h = max(1, height // divisor)
    if small_w == width and small_h == height:
        return region.copy()
    shrunk = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_LINEAR)


def _pixelate(region: np.ndarray, pixel_size: int) -> np.ndarray:
    height, width = region.shape[:2]
    block = max(2, pixel_size)
    small_w = max(1, width // block)
    small_h = max(1, height // block)
    shrunk = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_NEAREST)
