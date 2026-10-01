from pathlib import Path

import numpy as np
from PIL import Image

from noface.gallery import enroll
from noface.pipeline import process_frame
from noface.tracker import Tracker
from noface.types import Detection, Gallery, Options, Stats
from tests.fakes import ScriptedEngine

SUBJECT_BOX = (10.0, 10.0, 40.0, 50.0)
STRANGER_BOX = (80.0, 10.0, 120.0, 50.0)
SUBJECT_VECTOR = np.array([1.0, 0.0], np.float32)
STRANGER_VECTOR = np.array([0.0, 1.0], np.float32)


def _options(**overrides: object) -> Options:
    values: dict[str, object] = dict(
        style="black",
        pad=0.0,
        restore_pad=0.0,
        confirm_threshold=0.5,
        gray_threshold=0.28,
        track_ttl=2,
        hold_frames=2,
        max_gap=5,
        min_size=1,
        det_score=0.5,
    )
    values.update(overrides)
    return Options(**values)  # type: ignore[arg-type]


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "subject.jpg"
    Image.new("RGB", (16, 16), "white").save(path)
    return path


def _frame(value: int) -> np.ndarray:
    return np.full((80, 160, 3), value, np.uint8)


def test_subject_stays_and_stranger_is_blacked_out(tmp_path: Path) -> None:
    subject = Detection(SUBJECT_BOX, 0.99)
    stranger = Detection(STRANGER_BOX, 0.99)
    enrolled = Detection(SUBJECT_BOX, 0.99)
    engine = ScriptedEngine(
        [
            [(enrolled, SUBJECT_VECTOR)],
            [(subject, SUBJECT_VECTOR), (stranger, STRANGER_VECTOR)],
        ]
    )
    gallery = enroll([_reference(tmp_path)], engine, det_score=0.5).gallery
    stats = Stats()
    output = process_frame(_frame(200), engine, Tracker(), gallery, _options(), stats)
    assert np.all(output[10:50, 10:40] == 200)
    assert np.all(output[10:50, 80:120] == 0)
    assert stats.kept_confirm == 1
    assert stats.anonymized == 1


def test_confirmed_track_survives_a_weak_frame_then_a_miss(tmp_path: Path) -> None:
    enrolled = Detection(SUBJECT_BOX, 0.99)
    subject = Detection(SUBJECT_BOX, 0.99)
    turned = Detection(SUBJECT_BOX, 0.99)
    stranger = Detection(STRANGER_BOX, 0.99)
    stranger_again = Detection(STRANGER_BOX, 0.99)
    engine = ScriptedEngine(
        [
            [(enrolled, SUBJECT_VECTOR)],
            [(subject, SUBJECT_VECTOR), (stranger, STRANGER_VECTOR)],
            [(turned, STRANGER_VECTOR), (stranger_again, STRANGER_VECTOR)],
            [],
        ]
    )
    gallery = enroll([_reference(tmp_path)], engine, det_score=0.5).gallery
    tracker = Tracker(iou_threshold=0.3, max_missed=5)
    options = _options()
    stats = Stats()

    output = process_frame(_frame(200), engine, tracker, gallery, options, stats)
    assert np.all(output[10:50, 10:40] == 200)
    assert np.all(output[10:50, 80:120] == 0)

    output = process_frame(_frame(180), engine, tracker, gallery, options, stats)
    assert np.all(output[10:50, 10:40] == 180)
    assert np.all(output[10:50, 80:120] == 0)
    assert stats.kept_tracked == 1

    output = process_frame(_frame(150), engine, tracker, gallery, options, stats)
    assert np.all(output[10:50, 10:40] == 150)
    assert np.all(output[10:50, 80:120] == 0)
    assert stats.held_boxes == 1


def test_gray_match_does_not_protect_the_next_frame() -> None:
    gray = np.array([0.3, np.sqrt(1 - 0.3**2)], np.float32)
    orthogonal = np.array([0.0, 1.0], np.float32)
    gallery = Gallery(features=np.array([[1.0, 0.0]], np.float32))
    first = Detection(SUBJECT_BOX, 0.99)
    second = Detection(SUBJECT_BOX, 0.99)
    engine = ScriptedEngine([[(first, gray)], [(second, orthogonal)]])
    tracker = Tracker()
    stats = Stats()
    options = _options(track_ttl=4)
    output = process_frame(_frame(90), engine, tracker, gallery, options, stats)
    assert np.all(output[10:50, 10:40] == 90)
    assert stats.kept_gray == 1
    output = process_frame(_frame(90), engine, tracker, gallery, options, stats)
    assert np.all(output[10:50, 10:40] == 0)
    assert stats.anonymized == 1


def test_duplicate_detection_does_not_cover_the_subject(tmp_path: Path) -> None:
    enrolled = Detection(SUBJECT_BOX, 0.99)
    subject = Detection(SUBJECT_BOX, 0.99)
    duplicate = Detection((14.0, 14.0, 44.0, 54.0), 0.99)
    engine = ScriptedEngine(
        [
            [(enrolled, SUBJECT_VECTOR)],
            [(subject, SUBJECT_VECTOR), (duplicate, STRANGER_VECTOR)],
        ]
    )
    gallery = enroll([_reference(tmp_path)], engine, det_score=0.5).gallery
    output = process_frame(_frame(200), engine, Tracker(), gallery, _options(), Stats())
    assert np.all(output[10:54, 10:44] == 200)
