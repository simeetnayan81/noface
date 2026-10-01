"""Run enrollment, tracking, and anonymization over a video."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from noface.anonymize import apply_anonymization
from noface.errors import GalleryMismatch, NofaceError
from noface.gallery import enroll, log_enrollment, reference_images
from noface.geometry import box_is_large_enough, iou
from noface.match import decide, max_similarity
from noface.tracker import Tracker
from noface.types import FaceEngine, Gallery, Options, Stats
from noface.videoio import open_video, write_output

logger = logging.getLogger("noface")

# A second box on the same person (the two detector scales, or a held box)
# must not anonymize someone we have already decided to keep.
_SAME_FACE_IOU = 0.3


def anonymize_video(
    video_path: Path,
    reference_paths: list[Path],
    output_path: Path,
    engine: FaceEngine,
    options: Options,
    *,
    max_frames: int | None = None,
    keep_audio: bool = True,
) -> Stats:
    """Write ``output_path``, keeping every enrolled face and anonymizing the rest."""

    if video_path.resolve() == output_path.resolve():
        raise NofaceError("Choose an output path that is different from the input video.")
    if not video_path.is_file():
        raise NofaceError(f"Video not found: {video_path}")

    images = reference_images(reference_paths)
    enrollment = enroll(images, engine, det_score=options.det_score)
    log_enrollment(enrollment)

    source = open_video(video_path)
    stats = Stats(reference_images=len(images), enrolled_faces=enrollment.gallery.count)
    tracker = Tracker(iou_threshold=options.track_iou, max_missed=options.max_gap)
    started = time.perf_counter()
    limit = max_frames if max_frames and max_frames > 0 else None
    try:
        count_text = f", about {source.info.frame_count} frames" if source.info.frame_count else ""
        logger.info(
            "%s (%dx%d, %.3f fps%s)",
            video_path.name,
            source.info.width,
            source.info.height,
            source.info.fps,
            count_text,
        )
        total = source.info.frame_count
        if limit is not None and total is not None:
            total = min(total, limit)
        elif limit is not None:
            total = limit
        with tempfile.TemporaryDirectory(prefix="noface-") as temp_dir:
            silent_path = Path(temp_dir) / "video.mp4"
            writer = cv2.VideoWriter(
                str(silent_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source.info.fps,
                (source.info.width, source.info.height),
            )
            if not writer.isOpened():
                raise NofaceError("Could not start writing the output video.")
            progress = tqdm(total=total, unit="frame", desc="Anonymizing")
            try:
                for frame in source:
                    if limit is not None and stats.frames >= limit:
                        break
                    frame = _fit_frame(frame, source.info.width, source.info.height)
                    output = process_frame(
                        frame,
                        engine,
                        tracker,
                        enrollment.gallery,
                        options,
                        stats,
                    )
                    writer.write(output)
                    stats.frames += 1
                    progress.update(1)
            finally:
                progress.close()
                writer.release()
            if stats.frames == 0:
                detail = source.error_message()
                suffix = f" {detail}" if detail else ""
                raise NofaceError(f"No frames could be read from {video_path}.{suffix}")
            write_output(silent_path, video_path, output_path, keep_audio=keep_audio)
    finally:
        source.close()

    elapsed = time.perf_counter() - started
    logger.info(format_summary(stats, output_path, elapsed))
    return stats


def process_frame(
    frame: np.ndarray,
    engine: FaceEngine,
    tracker: Tracker,
    gallery: Gallery,
    options: Options,
    stats: Stats,
) -> np.ndarray:
    """Anonymize one BGR frame. ``tracker`` and ``stats`` are updated in place."""

    detections = [
        detection
        for detection in engine.detect(frame)
        if detection.score >= options.det_score
        and box_is_large_enough(detection.bbox, options.min_size)
    ]
    associations, held = tracker.match([detection.bbox for detection in detections])
    hide_boxes: list[tuple[float, float, float, float]] = []
    keep_boxes: list[tuple[float, float, float, float]] = []
    stats.detections += len(detections)

    for association in associations:
        detection = detections[association.det_index]
        try:
            similarity = max_similarity(engine.embed(frame, detection), gallery.features)
        except GalleryMismatch:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad face should be hidden, not abort the video
            logger.warning("Could not recognize a face, so it will be anonymized: %s", exc)
            similarity = -1.0
        keep, reason, new_ttl = decide(
            similarity,
            association.keep_ttl,
            confirm_threshold=options.confirm_threshold,
            gray_threshold=options.gray_threshold,
            track_ttl=options.track_ttl,
        )
        tracker.set_ttl(association.track_id, new_ttl)
        if keep:
            keep_boxes.append(detection.bbox)
            if reason == "confirm":
                stats.kept_confirm += 1
            elif reason == "gray":
                stats.kept_gray += 1
            else:
                stats.kept_tracked += 1
        else:
            hide_boxes.append(detection.bbox)

    hold_boxes: list[tuple[float, float, float, float]] = []
    for hold in held:
        if hold.keep_ttl > 0 or hold.missed > options.hold_frames:
            continue
        hold_boxes.append(hold.bbox)

    def _overlaps_kept(box: tuple[float, float, float, float]) -> bool:
        return any(iou(box, kept) >= _SAME_FACE_IOU for kept in keep_boxes)

    painted_hides = [box for box in hide_boxes if not _overlaps_kept(box)]
    painted_holds = [box for box in hold_boxes if not _overlaps_kept(box)]
    stats.anonymized += len(painted_hides)
    stats.held_boxes += len(painted_holds)
    return apply_anonymization(frame, painted_hides + painted_holds, keep_boxes, options)


def format_summary(stats: Stats, output: Path, elapsed: float) -> str:
    speed = stats.frames / elapsed if elapsed > 0 else 0.0
    lines = [
        f"Wrote {output}",
        f"{stats.frames} frames in {elapsed:.1f}s ({speed:.1f} fps)",
        (
            f"Kept face instances: {stats.kept} "
            f"(confirmed {stats.kept_confirm}, uncertain {stats.kept_gray}, "
            f"tracked {stats.kept_tracked})"
        ),
        f"Anonymized face instances: {stats.anonymized}",
    ]
    if stats.held_boxes:
        lines.append(f"Anonymized boxes held across missed detections: {stats.held_boxes}")
    return "\n".join(lines)


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[0] < height or frame.shape[1] < width or frame.shape[2] != 3:
        raise NofaceError(f"Unexpected frame shape {tuple(frame.shape)} for a {width}x{height} video.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[:height, :width]
