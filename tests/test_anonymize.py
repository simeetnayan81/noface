import numpy as np

from noface.anonymize import apply_anonymization
from noface.types import Options


def _frame() -> np.ndarray:
    frame = np.full((80, 120, 3), 180, np.uint8)
    frame[0, 0] = (1, 2, 3)
    return frame


def test_black_box_changes_only_the_region() -> None:
    frame = _frame()
    output = apply_anonymization(frame, [(10, 20, 40, 50)], [], Options(style="black", pad=0.0))
    assert np.all(output[20:50, 10:40] == 0)
    assert np.all(output[0, 0] == frame[0, 0])
    assert np.all(output[55:70, 50:80] == 180)
    assert frame[20, 10, 0] == 180
    assert tuple(int(v) for v in frame[0, 0]) == (1, 2, 3)


def test_blur_reduces_detail_inside_the_box() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(90, 90, 3), dtype=np.uint8)
    sentinel = frame[0, 0].copy()
    output = apply_anonymization(
        frame,
        [(10, 10, 80, 80)],
        [],
        Options(style="blur", blur_strength=16, pad=0.0),
    )
    assert np.all(output[0, 0] == sentinel)
    original_std = float(frame[10:80, 10:80].std())
    blurred_std = float(output[10:80, 10:80].std())
    assert blurred_std < original_std / 2


def test_pixelate_makes_constant_blocks() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[:, :, 0] = np.arange(40, dtype=np.uint8)
    output = apply_anonymization(
        frame,
        [(0, 0, 40, 40)],
        [],
        Options(style="pixelate", pixel_size=10, pad=0.0),
    )
    block = output[0:10, 0:10]
    assert np.all(block == block[0, 0])


def test_kept_face_is_restored_when_boxes_overlap() -> None:
    frame = np.full((100, 160, 3), 210, np.uint8)
    subject = (40.0, 30.0, 100.0, 90.0)
    stranger = (80.0, 30.0, 140.0, 90.0)
    output = apply_anonymization(
        frame,
        [stranger],
        [subject],
        Options(style="black", pad=0.0, restore_pad=0.0),
    )
    assert np.all(output[30:90, 40:100] == 210)
    assert np.all(output[30:90, 100:140] == 0)
