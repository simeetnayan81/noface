"""Identity scores and the keep / anonymize decision."""

from __future__ import annotations

import math

import numpy as np

from noface.errors import GalleryMismatch, NofaceError


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12 or not math.isfinite(norm):
        raise NofaceError("Face embedding was empty.")
    return vector / norm


def max_similarity(feature: np.ndarray, gallery: np.ndarray) -> float:
    """Highest cosine similarity between ``feature`` and any kept face.

    Both sides are L2-normalized, so the dot product is cosine similarity.
    SFace's raw vectors are not unit length; normalizing them matches
    ``FaceRecognizerSF.match(..., FR_COSINE)``.
    """

    vector = l2_normalize(feature)
    if gallery.ndim != 2 or gallery.shape[1] != vector.shape[0]:
        width = gallery.shape[1] if gallery.ndim == 2 else "?"
        raise GalleryMismatch(
            f"Embedding length {vector.shape[0]} does not match the reference gallery ({width})."
        )
    return float(np.max(gallery @ vector))


def decide(
    similarity: float,
    keep_ttl: int,
    *,
    confirm_threshold: float,
    gray_threshold: float,
    track_ttl: int,
) -> tuple[bool, str, int]:
    """Choose whether a face stays visible.

    A score at or above ``confirm_threshold`` keeps the face and remembers the
    track for ``track_ttl`` further weak sightings. A score in the gray band
    stays visible for this frame and does not refresh that memory. Below the
    gray band, the face stays visible only while the track still has memory
    left. Otherwise it is anonymized.

    Returns ``(keep, reason, new_keep_ttl)``. ``reason`` is ``confirm``,
    ``gray``, ``tracked``, or ``anonymize``.
    """

    if not math.isfinite(similarity):
        similarity = -1.0
    if similarity >= confirm_threshold:
        return True, "confirm", max(0, track_ttl)
    if similarity >= gray_threshold:
        return True, "gray", max(0, keep_ttl)
    if keep_ttl > 0:
        return True, "tracked", keep_ttl - 1
    return False, "anonymize", 0
