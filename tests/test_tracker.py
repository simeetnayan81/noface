from noface.tracker import Tracker


def test_tracks_follow_motion_and_expire() -> None:
    tracker = Tracker(iou_threshold=0.3, max_missed=2)
    left = (0.0, 0.0, 10.0, 10.0)
    right = (50.0, 0.0, 60.0, 10.0)
    first, held = tracker.match([left, right])
    assert held == []
    ids = {item.det_index: item.track_id for item in first}
    tracker.set_ttl(ids[0], 5)

    second, held = tracker.match([(1.0, 0.0, 11.0, 10.0), (51.0, 0.0, 61.0, 10.0)])
    assert held == []
    assert {item.det_index: item.track_id for item in second} == ids
    assert next(item.keep_ttl for item in second if item.det_index == 0) == 5

    _, held = tracker.match([])
    assert [item.track_id for item in held] == [ids[0], ids[1]]
    assert held[0].missed == 1
    assert held[0].keep_ttl == 5

    _, held = tracker.match([])
    assert held[0].missed == 2
    _, held = tracker.match([])
    assert held == []

    restarted, _ = tracker.match([left])
    assert restarted[0].track_id not in ids.values()
    assert restarted[0].keep_ttl == 0


def test_distant_detection_does_not_steal_a_track() -> None:
    tracker = Tracker(iou_threshold=0.5, max_missed=5)
    tracker.match([(0.0, 0.0, 10.0, 10.0)])
    associations, held = tracker.match([(100.0, 100.0, 110.0, 110.0)])
    assert len(associations) == 1
    assert associations[0].track_id == 2
    assert [item.track_id for item in held] == [1]
