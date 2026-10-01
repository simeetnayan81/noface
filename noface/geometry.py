"""Boxes, overlap, and the detector scale ladder."""

from __future__ import annotations

import math

# YuNet is trained for faces of roughly 10–300 px in the image it actually sees.
# Stay inside a slightly tighter band so close-ups and tiny faces both land cleanly.
MIN_DETECTOR_FACE = 20.0
MAX_DETECTOR_FACE = 280.0


def iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def expand_box(
    box: tuple[float, float, float, float],
    pad: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Grow ``box`` by ``pad`` on each side and clip it to the frame.

    Returns integer ``(x1, y1, x2, y2)`` suitable for slicing, or ``None``
    when the clipped region is too small to draw.
    """

    x1, y1, x2, y2 = box
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    x1 -= box_w * pad
    x2 += box_w * pad
    y1 -= box_h * pad
    y2 += box_h * pad
    left = max(0, math.floor(x1))
    top = max(0, math.floor(y1))
    right = min(width, math.ceil(x2))
    bottom = min(height, math.ceil(y2))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def detection_scales(width: int, height: int, max_side: int = 1920) -> list[float]:
    """Scales that bring both distant faces and close-ups into YuNet's range.

    The detector sees faces of about 20–280 px reliably. One pass at (near)
    native resolution finds small faces. A second, smaller pass finds a face
    that fills the frame. ``max_side`` caps the larger pass so a 4K frame is
    not detected at full resolution.
    """

    if width < 1 or height < 1 or max_side < 1:
        return [1.0]
    long_side = max(width, height)
    short_side = min(width, height)
    upper = min(1.0, max_side / long_side)
    close = min(upper, MAX_DETECTOR_FACE / short_side)
    if close < upper * 0.85:
        return [upper, close]
    return [upper]


def box_is_large_enough(box: tuple[float, float, float, float], min_size: float) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width >= min_size and height >= min_size
