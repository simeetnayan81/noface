"""Load reference photos and enroll every face in them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from noface.errors import NofaceError
from noface.match import l2_normalize
from noface.types import EnrolledFace, FaceEngine, Gallery

logger = logging.getLogger("noface")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Enrollment:
    gallery: Gallery
    per_image: list[tuple[Path, int]]


def reference_images(paths: list[Path]) -> list[Path]:
    """Expand files and folders into a de-duplicated image list.

    A folder contributes every image inside it, including subfolders. Every
    face found in those images is someone to keep. Names that start with ``.``
    are skipped.
    """

    found: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                supported = ", ".join(sorted(IMAGE_SUFFIXES))
                raise NofaceError(f"{path} is not a supported image ({supported}).")
            found.append(path)
        elif path.is_dir():
            files = [
                item
                for item in path.rglob("*")
                if item.is_file()
                and item.suffix.lower() in IMAGE_SUFFIXES
                and not _is_hidden(path, item)
            ]
            if not files:
                raise NofaceError(f"No images found in {path}.")
            found.extend(files)
        else:
            raise NofaceError(f"Reference not found: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for item in sorted(found, key=lambda candidate: str(candidate).lower()):
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(item)
    return unique


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR, applying EXIF orientation from phone cameras."""

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise NofaceError(f"Could not read image {path}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise NofaceError(f"Could not read a color image from {path}.")
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def enroll(images: list[Path], engine: FaceEngine, *, det_score: float) -> Enrollment:
    """Build a gallery from every face in ``images``."""

    features: list[np.ndarray] = []
    faces: list[EnrolledFace] = []
    per_image: list[tuple[Path, int]] = []
    for path in images:
        image = read_image(path)
        detections = [det for det in engine.detect(image) if det.score >= det_score]
        per_image.append((path, len(detections)))
        for detection in detections:
            try:
                feature = l2_normalize(engine.embed(image, detection))
            except Exception as exc:
                raise NofaceError(f"Could not embed a face in {path}: {exc}") from exc
            features.append(feature)
            faces.append(EnrolledFace(source=str(path), score=float(detection.score)))

    if not features:
        raise NofaceError(
            "No faces found in the reference images. "
            "Use a clear photo of each person whose face should stay visible."
        )
    gallery = Gallery(features=np.stack(features).astype(np.float32), faces=faces)
    return Enrollment(gallery=gallery, per_image=per_image)


def log_enrollment(enrollment: Enrollment) -> None:
    for path, count in enrollment.per_image:
        if count == 0:
            logger.warning("%s: no face detected", path)
        elif count == 1:
            logger.info("%s: 1 face to keep", path)
        else:
            logger.info(
                "%s: %d faces to keep (everyone detected in this image stays visible)",
                path,
                count,
            )
    images = len(enrollment.per_image)
    faces = enrollment.gallery.count
    logger.info(
        "Enrolled %d face%s from %d image%s.",
        faces,
        "" if faces == 1 else "s",
        images,
        "" if images == 1 else "s",
    )


def _is_hidden(root: Path, item: Path) -> bool:
    relative = item.relative_to(root)
    return any(part.startswith(".") for part in relative.parts)
