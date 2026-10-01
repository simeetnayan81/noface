from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from noface.errors import NofaceError
from noface.gallery import enroll, read_image, reference_images
from noface.types import Detection
from tests.fakes import ScriptedEngine


def _save(path: Path, size: tuple[int, int] = (16, 16), color: tuple[int, int, int] = (255, 255, 255)) -> None:
    Image.new("RGB", size, color).save(path)


def test_reference_listing_skips_hidden_and_non_images(tmp_path: Path) -> None:
    folder = tmp_path / "people"
    nested = folder / "more"
    hidden = folder / ".secret"
    nested.mkdir(parents=True)
    hidden.mkdir()
    _save(folder / "ada.jpg")
    _save(nested / "bea.png")
    _save(hidden / "skip.jpg")
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    found = reference_images([folder])
    assert [path.name for path in found] == ["ada.jpg", "bea.png"]


def test_reference_file_and_missing(tmp_path: Path) -> None:
    photo = tmp_path / "ada.jpg"
    _save(photo)
    assert reference_images([photo]) == [photo]
    (tmp_path / "notes.txt").write_text("no", encoding="utf-8")
    with pytest.raises(NofaceError):
        reference_images([tmp_path / "notes.txt"])
    with pytest.raises(NofaceError):
        reference_images([tmp_path / "missing"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NofaceError):
        reference_images([empty])


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    image = Image.new("RGB", (30, 10), "black")
    image.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6  # display rotated 90 degrees clockwise
    path = tmp_path / "turned.jpg"
    image.save(path, exif=exif)
    bgr = read_image(path)
    assert bgr.shape[0] == 30
    assert bgr.shape[1] == 10


def test_enroll_keeps_every_face_in_an_image(tmp_path: Path) -> None:
    photo = tmp_path / "group.jpg"
    _save(photo)
    first = Detection((1.0, 1.0, 10.0, 12.0), 0.9)
    second = Detection((20.0, 4.0, 40.0, 30.0), 0.8)
    weak = Detection((50.0, 4.0, 70.0, 30.0), 0.2)
    engine = ScriptedEngine(
        [
            [
                (first, np.array([1.0, 0.0], np.float32)),
                (second, np.array([0.0, 1.0], np.float32)),
                (weak, np.array([0.0, 0.0], np.float32)),
            ]
        ]
    )
    enrollment = enroll([photo], engine, det_score=0.6)
    assert enrollment.per_image == [(photo, 2)]
    assert enrollment.gallery.count == 2
    assert enrollment.gallery.features.shape == (2, 2)


def test_enroll_fails_when_no_face_is_found(tmp_path: Path) -> None:
    photo = tmp_path / "wall.jpg"
    _save(photo)
    engine = ScriptedEngine([[]])
    with pytest.raises(NofaceError):
        enroll([photo], engine, det_score=0.6)
