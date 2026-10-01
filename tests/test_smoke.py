"""Real-model check. Skipped unless NOFACE_SMOKE=1.

Run it with the face models available:

    NOFACE_SMOKE=1 pytest tests/test_smoke.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from noface.engine import OpenCVFaceEngine
from noface.pipeline import anonymize_video
from noface.types import Options
from noface.videoio import FFmpegVideoSource

pytestmark = pytest.mark.skipif(
    os.environ.get("NOFACE_SMOKE") != "1",
    reason="set NOFACE_SMOKE=1 to run YuNet and SFace on a real frame",
)

LENA_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
MESSI_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg"
SAMPLE_DIR = Path("/tmp/noface-samples")
MODEL_DIR = Path("/tmp/noface-models")


def _sample(name: str, url: str) -> Path:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DIR / name
    if not path.is_file() or path.stat().st_size < 1000:
        request = urllib.request.Request(url, headers={"User-Agent": "noface-test"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _first_frame(path: Path) -> np.ndarray:
    source = FFmpegVideoSource(path)
    try:
        return next(iter(source))
    finally:
        source.close()


def _video_from_canvas(canvas_path: Path, video_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(canvas_path),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-frames:v",
            "6",
            "-r",
            "10",
            "-shortest",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video_path),
        ],
        check=True,
    )


def test_lena_is_kept_and_messi_is_anonymized(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    import cv2

    lena_path = _sample("lena.jpg", LENA_URL)
    messi_path = _sample("messi5.jpg", MESSI_URL)
    lena = cv2.imread(str(lena_path))
    messi = cv2.imread(str(messi_path))
    assert lena is not None and messi is not None

    canvas = np.full((720, 1280, 3), 30, np.uint8)
    canvas[80 : 80 + lena.shape[0], 40 : 40 + lena.shape[1]] = lena
    canvas[160 : 160 + messi.shape[0], 700 : 700 + messi.shape[1]] = messi
    canvas_path = tmp_path / "canvas.png"
    cv2.imwrite(str(canvas_path), canvas)

    engine = OpenCVFaceEngine(model_dir=MODEL_DIR, det_score=0.6, max_det_side=1280)
    faces = engine.detect(canvas)

    def center(box: tuple[float, float, float, float]) -> tuple[int, int]:
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def inside(box: tuple[float, float, float, float], origin_x: int, origin_y: int, image: np.ndarray) -> bool:
        x, y = center(box)
        return origin_x <= x < origin_x + image.shape[1] and origin_y <= y < origin_y + image.shape[0]

    lena_faces = [face for face in faces if inside(face.bbox, 40, 80, lena)]
    messi_faces = [face for face in faces if inside(face.bbox, 700, 160, messi)]
    assert lena_faces, "Lena's face was not detected on the canvas"
    assert messi_faces, "the second face was not detected on the canvas"

    from noface.gallery import enroll
    from noface.match import max_similarity

    gallery = enroll([lena_path], engine, det_score=0.6).gallery
    lena_score = max(max_similarity(engine.embed(canvas, face), gallery.features) for face in lena_faces)
    messi_score = max(max_similarity(engine.embed(canvas, face), gallery.features) for face in messi_faces)
    assert lena_score > 0.5
    assert messi_score < 0.363
    kept_box = max(lena_faces, key=lambda face: face.score).bbox
    hidden_box = max(messi_faces, key=lambda face: face.score).bbox

    video = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    _video_from_canvas(canvas_path, video)
    anonymize_video(
        video,
        [lena_path],
        output,
        engine,
        Options(style="black", pad=0.2, det_score=0.6, min_size=12, max_det_side=1280),
    )
    original = _first_frame(video)
    result = _first_frame(output)

    def patch(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        x, y = center(box)
        return image[y - 6 : y + 6, x - 6 : x + 6]

    kept_delta = np.abs(patch(original, kept_box).astype(np.int16) - patch(result, kept_box).astype(np.int16))
    assert float(kept_delta.mean()) < 25
    assert float(patch(result, hidden_box).mean()) < 20

    both = tmp_path / "both.mp4"
    anonymize_video(
        video,
        [canvas_path],
        both,
        engine,
        Options(style="black", pad=0.2, det_score=0.6, min_size=12, max_det_side=1280),
    )
    kept_both = _first_frame(both)
    # A reference frame that contains both people keeps both faces.
    both_delta = np.abs(
        patch(original, hidden_box).astype(np.int16) - patch(kept_both, hidden_box).astype(np.int16)
    )
    assert float(both_delta.mean()) < 25
