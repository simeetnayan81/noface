from pathlib import Path

from PIL import Image

from noface.cli import build_parser, main


def test_help_mentions_references() -> None:
    help_text = build_parser().format_help()
    assert "people to keep" in help_text
    assert "--style" in help_text
    assert "--gray-threshold" in help_text


def test_missing_video_is_rejected_before_models_download(tmp_path: Path) -> None:
    photo = tmp_path / "face.jpg"
    Image.new("RGB", (8, 8), "white").save(photo)
    code = main(
        [
            "--video",
            str(tmp_path / "missing.mp4"),
            "--references",
            str(photo),
            "--output",
            str(tmp_path / "out.mp4"),
        ]
    )
    assert code == 1


def test_threshold_order_is_rejected(tmp_path: Path) -> None:
    photo = tmp_path / "face.jpg"
    Image.new("RGB", (8, 8), "white").save(photo)
    code = main(
        [
            "--references",
            str(photo),
            "--dry-run",
            "--gray-threshold",
            "0.9",
            "--confirm-threshold",
            "0.2",
        ]
    )
    assert code == 1


def test_output_cannot_replace_the_input(tmp_path: Path) -> None:
    photo = tmp_path / "face.jpg"
    video = tmp_path / "clip.mp4"
    Image.new("RGB", (8, 8), "white").save(photo)
    video.write_bytes(b"not a video")
    code = main(
        [
            "--video",
            str(video),
            "--references",
            str(photo),
            "--output",
            str(video),
        ]
    )
    assert code == 1
