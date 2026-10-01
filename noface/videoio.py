"""Read video frames and write a file that keeps the original audio.

ffmpeg is used when it is installed: it honors phone-video rotation and can
mux the original audio back on. OpenCV is the fallback reader and always
writes the silent intermediate.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from noface.errors import NofaceError

logger = logging.getLogger("noface")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None
    rotation: float = 0.0


class FFmpegVideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        coded_w, coded_h, fps, frame_count, rotation = probe_video(path)
        self.display_width, self.display_height = display_dimensions(coded_w, coded_h, rotation)
        width, height = even_dimensions(self.display_width, self.display_height)
        if width < 2 or height < 2:
            raise NofaceError(f"{path} has no usable video frames.")
        self.info = VideoInfo(width, height, fps, frame_count, rotation)
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr: list[str] = []

    def __enter__(self) -> FFmpegVideoSource:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[np.ndarray]:
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.path),
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_popen_kwargs(),
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stderr_thread = threading.Thread(
            target=_drain_text,
            args=(self._proc.stderr, self._stderr),
            daemon=True,
        )
        self._stderr_thread.start()
        frame_bytes = self.display_width * self.display_height * 3
        while True:
            payload = _read_exact(self._proc.stdout, frame_bytes)
            if payload is None:
                break
            if len(payload) != frame_bytes:
                break
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                self.display_height, self.display_width, 3
            )
            yield frame[: self.info.height, : self.info.width].copy()
        self._finish_process()

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

    def error_message(self) -> str | None:
        proc = self._proc
        if proc is None or proc.returncode in (0, None):
            return None
        text = "".join(self._stderr).strip()
        return text or f"ffmpeg exited with status {proc.returncode}"

    def _finish_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait(timeout=30)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)


class OpenCVVideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise NofaceError(f"Could not open video {path}.")
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        even_w, even_h = even_dimensions(width, height)
        if even_w < 2 or even_h < 2:
            self._cap.release()
            raise NofaceError(f"Could not read frames from {path}.")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1 or fps > 480:
            fps = 30.0
        count_value = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.info = VideoInfo(even_w, even_h, fps, count_value or None, 0.0)

    def __enter__(self) -> OpenCVVideoSource:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[np.ndarray]:
        width, height = self.info.width, self.info.height
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            yield frame[:height, :width].copy()

    def close(self) -> None:
        self._cap.release()

    def error_message(self) -> str | None:
        return None


def open_video(path: Path) -> FFmpegVideoSource | OpenCVVideoSource:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return FFmpegVideoSource(path)
    logger.warning(
        "ffmpeg is not installed. Audio will be dropped, and phone videos may open sideways."
    )
    return OpenCVVideoSource(path)


def write_output(silent_video: Path, audio_source: Path, output: Path, *, keep_audio: bool) -> None:
    """Encode ``silent_video`` as H.264 and, when possible, copy ``audio_source`` audio."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        shutil.move(str(silent_video), output)
        logger.warning(
            "ffmpeg is not installed, so %s has no audio and may not play in every app.",
            output,
        )
        return

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
    ]
    if keep_audio:
        command.extend(["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        command.extend(["-map", "0:v:0"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if keep_audio:
        command.extend(["-c:a", "aac", "-shortest"])
    else:
        command.append("-an")
    command.append(str(output))
    completed = subprocess.run(command, capture_output=True, text=True, **_popen_kwargs())
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        detail = completed.stderr.strip() or f"ffmpeg exited with status {completed.returncode}"
        raise NofaceError(f"ffmpeg could not write {output}: {detail}")


def probe_video(path: Path) -> tuple[int, int, float, int | None, float]:
    """Return coded width, coded height, fps, frame count, and rotation in degrees."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, **_popen_kwargs())
    except FileNotFoundError as exc:
        raise NofaceError("ffprobe is not installed.") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or "ffprobe failed"
        raise NofaceError(f"Could not read video {path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise NofaceError(f"Could not read video {path}: {exc}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise NofaceError(f"No video stream in {path}.")
    return parse_stream(streams[0])


def parse_stream(stream: dict) -> tuple[int, int, float, int | None, float]:
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NofaceError("The video probe did not include a frame size.") from exc
    fps = parse_frame_rate(stream.get("avg_frame_rate"))
    if fps < 1 or fps > 480:
        fps = parse_frame_rate(stream.get("r_frame_rate"))
    if fps < 1 or fps > 480:
        fps = 30.0
    rotation = stream_rotation(stream)
    return width, height, fps, frame_count_of(stream, fps), rotation


def parse_frame_rate(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value)
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num_text, den_text = text.split("/", 1)
        try:
            numerator = float(num_text)
            denominator = float(den_text)
        except ValueError:
            return 0.0
        if denominator == 0:
            return 0.0
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return 0.0


def stream_rotation(stream: dict) -> float:
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and item.get("rotation") is not None:
            try:
                return float(item["rotation"])
            except (TypeError, ValueError):
                continue
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is None:
        return 0.0
    try:
        return float(tag)
    except (TypeError, ValueError):
        return 0.0


def frame_count_of(stream: dict, fps: float) -> int | None:
    nb_frames = stream.get("nb_frames")
    if isinstance(nb_frames, str) and nb_frames.isdigit() and int(nb_frames) > 0:
        return int(nb_frames)
    duration = stream.get("duration")
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        return None
    if seconds > 0 and fps > 0:
        return max(1, int(round(seconds * fps)))
    return None


def display_dimensions(width: int, height: int, rotation: float) -> tuple[int, int]:
    """Size ffmpeg will emit after applying a 90-degree display rotation."""

    turns = int(round(abs(rotation))) % 360
    if turns in (90, 270):
        return height, width
    return width, height


def even_dimensions(width: int, height: int) -> tuple[int, int]:
    return width - (width % 2), height - (height % 2)


def _read_exact(pipe: object, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    read = getattr(pipe, "read")
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            return b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain_text(pipe: object, bucket: list[str]) -> None:
    try:
        for line in iter(getattr(pipe, "readline"), b""):
            bucket.append(line.decode("utf-8", "replace"))
            if len(bucket) > 40:
                del bucket[:-40]
    except Exception:  # noqa: BLE001 - the reader should not die on a log line
        return


def _popen_kwargs() -> dict:
    if os.name == "nt":
        flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flag}
    return {}
