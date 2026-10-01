"""IoU tracker that carries a confirmed identity across nearby frames."""

from __future__ import annotations

from dataclasses import dataclass

from noface.geometry import iou


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    keep_ttl: int = 0
    missed: int = 0
    hits: int = 1


@dataclass(frozen=True)
class Association:
    """A detection matched to a track. ``keep_ttl`` is the value before this frame."""

    det_index: int
    track_id: int
    keep_ttl: int


@dataclass(frozen=True)
class HeldTrack:
    """A track with no detection this frame, still inside the gap window."""

    track_id: int
    bbox: tuple[float, float, float, float]
    keep_ttl: int
    missed: int


class Tracker:
    """Greedy IoU tracker.

    A confirmed subject keeps ``keep_ttl`` while the track is alive, including
    across a few missed frames, so a turned or blocked face is not anonymized
    the moment the embedding dips. ``max_missed`` is how many consecutive
    misses the same track can absorb before it is forgotten.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15) -> None:
        if iou_threshold <= 0:
            raise ValueError("iou_threshold must be positive")
        if max_missed < 0:
            raise ValueError("max_missed cannot be negative")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def match(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> tuple[list[Association], list[HeldTrack]]:
        track_ids = list(self._tracks)
        pairs: list[tuple[float, int, int]] = []
        for track_index, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for det_index, box in enumerate(boxes):
                overlap = iou(track.bbox, box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_index, det_index))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        associations: list[Association] = []
        for _overlap, track_index, det_index in pairs:
            if track_index in matched_tracks or det_index in matched_dets:
                continue
            track = self._tracks[track_ids[track_index]]
            track.bbox = boxes[det_index]
            track.missed = 0
            track.hits += 1
            matched_tracks.add(track_index)
            matched_dets.add(det_index)
            associations.append(
                Association(
                    det_index=det_index,
                    track_id=track.track_id,
                    keep_ttl=track.keep_ttl,
                )
            )

        held: list[HeldTrack] = []
        drop: list[int] = []
        for track_index, track_id in enumerate(track_ids):
            if track_index in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                drop.append(track_id)
                continue
            held.append(
                HeldTrack(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    keep_ttl=track.keep_ttl,
                    missed=track.missed,
                )
            )
        for track_id in drop:
            del self._tracks[track_id]

        for det_index, box in enumerate(boxes):
            if det_index in matched_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(track_id=track_id, bbox=box)
            associations.append(Association(det_index=det_index, track_id=track_id, keep_ttl=0))

        associations.sort(key=lambda item: item.det_index)
        return associations, held

    def set_ttl(self, track_id: int, keep_ttl: int) -> None:
        self._tracks[track_id].keep_ttl = max(0, keep_ttl)
