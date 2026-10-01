import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from noface.pipeline import anonymize_video
from noface.types import Detection, Options
from noface.videoio import (
    FFmpegVideoSource,
    display_dimensions,
    even_dimensions,
    parse_frame_rate,
    parse_stream,
)
from tests.fakes import ScriptedEngine

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def test_parse_stream_reads_rotation_rate_and_count() -> None:
    stream = {
        "width": 320,
        "height": 240,
        "avg_frame_rate": "30000/1001",
        "nb_frames": "15",
        "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
    }
    width, height, fps, count, rotation = parse_stream(stream)
    assert (width, height, count, rotation) == (320, 240, 15, -90.0)
    assert fps == pytest.approx(29.97, abs=0.01)
    assert display_dimensions(width, height, rotation) == (240, 320)
    assert display_dimensions(1920, 1080, 180) == (1920, 1080)
    assert even_dimensions(1919, 1080) == (1918, 1080)
    assert parse_frame_rate("0/0") == 0.0
    assert parse_frame_rate(None) == 0.0


def test_frame_count_falls_back_to_duration() -> None:
    _width, _height, _fps, count, _rotation = parse_stream(
        {"width": 10, "height": 8, "avg_frame_rate": "10/1", "nb_frames": "N/A", "duration": "1.5"}
    )
    assert count == 15


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")
def test_reader_applies_display_rotation(tmp_path: Path) -> None:
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(base),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    source = FFmpegVideoSource(rotated)
    try:
        frame = next(iter(source))
    finally:
        source.close()
    assert source.info.width == 240
    assert source.info.height == 320
    assert frame.shape == (320, 240, 3)
    # Blue in BGR is the first channel. Compression leaves it dominant.
    assert frame[:, :, 0].mean() > frame[:, :, 1].mean()
    assert frame[:, :, 0].mean() > frame[:, :, 2].mean()


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")
def test_video_roundtrip_keeps_audio_when_there_are_no_faces(tmp_path: Path) -> None:
    from PIL import Image

    video = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.jpg"
    Image.new("RGB", (16, 16), "white").save(reference)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:r=10:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:d=0.4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    face = Detection((0.0, 0.0, 8.0, 8.0), 0.99)
    engine = ScriptedEngine([[(face, np.array([1.0, 0.0], np.float32))]])
    stats = anonymize_video(
        video,
        [reference],
        output,
        engine,
        Options(style="black", min_size=1),
        keep_audio=True,
    )
    assert stats.frames >= 1
    assert output.is_file()
    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "audio" in probed.stdout
    source = FFmpegVideoSource(output)
    try:
        frame = next(iter(source))
    finally:
        source.close()
    assert frame.shape == (120, 160, 3)
    assert frame[:, :, 2].mean() > frame[:, :, 0].mean()
