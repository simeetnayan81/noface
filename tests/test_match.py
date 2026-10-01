import numpy as np
import pytest

from noface.errors import GalleryMismatch, NofaceError
from noface.match import decide, l2_normalize, max_similarity


def test_normalize_and_max_similarity() -> None:
    subject = l2_normalize(np.array([2.0, 0.0, 0.0]))
    stranger = l2_normalize(np.array([0.0, 5.0, 0.0]))
    gallery = np.stack([subject])
    assert max_similarity(subject * 4, gallery) == pytest.approx(1.0)
    assert max_similarity(stranger, gallery) == pytest.approx(0.0)
    assert max_similarity(np.array([1.0, 1.0, 0.0]), gallery) == pytest.approx(0.70710677, abs=1e-5)


def test_empty_embedding_and_mismatch() -> None:
    with pytest.raises(NofaceError):
        l2_normalize(np.array([0.0, 0.0]))
    with pytest.raises(GalleryMismatch):
        max_similarity(np.array([1.0, 0.0, 0.0]), np.ones((1, 2), dtype=np.float32))


def test_decide_confirm_gray_track_and_hide() -> None:
    keep, reason, ttl = decide(0.9, 0, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (True, "confirm", 3)

    # Gray stays visible and does not refresh the memory from the confirm above.
    keep, reason, ttl = decide(0.30, 3, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (True, "gray", 3)

    keep, reason, ttl = decide(0.10, 3, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (True, "tracked", 2)
    keep, reason, ttl = decide(0.10, 1, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (True, "tracked", 0)
    keep, reason, ttl = decide(0.10, 0, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (False, "anonymize", 0)

    # A gray score with no prior confirm does not start a track.
    keep, reason, ttl = decide(0.28, 0, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (True, "gray", 0)
    keep, reason, ttl = decide(float("nan"), 0, confirm_threshold=0.363, gray_threshold=0.28, track_ttl=3)
    assert (keep, reason, ttl) == (False, "anonymize", 0)
