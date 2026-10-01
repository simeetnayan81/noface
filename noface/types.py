"""Shared data types for detection, options, and run statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """One face in an image.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels. ``raw`` is the detector row
    (box, landmarks, score) in the same coordinate space, kept so the
    recognizer can align the face.
    """

    bbox: tuple[float, float, float, float]
    score: float
    raw: np.ndarray | None = None


class FaceEngine(Protocol):
    """Detects faces and turns each one into an identity embedding."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return faces in ``image`` (BGR uint8)."""

    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        """Return a feature vector for ``detection``."""


@dataclass
class Options:
    """Knobs for one anonymization run. Defaults match the CLI."""

    style: str = "blur"
    blur_strength: int = 16
    pixel_size: int = 12
    pad: float = 0.35
    restore_pad: float = 0.1
    confirm_threshold: float = 0.363
    gray_threshold: float = 0.28
    track_ttl: int = 30
    track_iou: float = 0.3
    max_gap: int = 15
    hold_frames: int = 4
    det_score: float = 0.6
    min_size: float = 12.0
    max_det_side: int = 1920


@dataclass
class EnrolledFace:
    source: str
    score: float


@dataclass
class Gallery:
    """L2-normalized embeddings of every face the user asked to keep."""

    features: np.ndarray
    faces: list[EnrolledFace] = field(default_factory=list)

    @property
    def count(self) -> int:
        if self.features.ndim != 2:
            return 0
        return int(self.features.shape[0])


@dataclass
class Stats:
    reference_images: int = 0
    enrolled_faces: int = 0
    frames: int = 0
    detections: int = 0
    kept_confirm: int = 0
    kept_gray: int = 0
    kept_tracked: int = 0
    anonymized: int = 0
    held_boxes: int = 0

    @property
    def kept(self) -> int:
        return self.kept_confirm + self.kept_gray + self.kept_tracked
