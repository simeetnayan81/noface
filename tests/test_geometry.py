from noface.geometry import (
    MAX_DETECTOR_FACE,
    MIN_DETECTOR_FACE,
    detection_scales,
    expand_box,
    iou,
)


def test_iou_identical_partial_and_separate() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0
    assert iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0
    # 10x10 overlapping a 10x10 shifted by 5 in x: intersection 50, union 150.
    assert abs(iou(box, (5.0, 0.0, 15.0, 10.0)) - (50 / 150)) < 1e-9


def test_expand_box_pads_and_clips() -> None:
    assert expand_box((10, 10, 20, 30), 0.0, 100, 80) == (10, 10, 20, 30)
    grown = expand_box((10, 10, 20, 30), 0.5, 100, 80)
    assert grown == (5, 0, 25, 40)
    assert expand_box((-30, -30, -10, -10), 0.0, 50, 50) is None


def test_detection_scales_cover_small_and_closeup_faces() -> None:
    cases = [(1920, 1080, 1920), (1080, 1920, 1920), (3840, 2160, 1920), (640, 480, 1920)]
    for width, height, max_side in cases:
        scales = detection_scales(width, height, max_side=max_side)
        assert scales
        assert all(0 < scale <= 1 for scale in scales)
        long_side = max(width, height)
        short_side = min(width, height)
        upper = min(1.0, max_side / long_side)
        smallest = MIN_DETECTOR_FACE / upper
        face = smallest
        while face <= short_side + 1e-6:
            assert any(
                MIN_DETECTOR_FACE - 1e-6 <= face * scale <= MAX_DETECTOR_FACE + 1e-6
                for scale in scales
            ), (width, height, face, scales)
            face *= 1.35


def test_small_image_uses_one_scale() -> None:
    assert detection_scales(200, 180, max_side=1920) == [1.0]
