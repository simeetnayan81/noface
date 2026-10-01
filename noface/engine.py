"""YuNet face detection and SFace identity embeddings via OpenCV."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from noface.errors import NofaceError
from noface.geometry import (
    MAX_DETECTOR_FACE,
    MIN_DETECTOR_FACE,
    detection_scales,
    iou,
)
from noface.types import Detection

logger = logging.getLogger("noface")

_YUNET_2026 = "face_detection_yunet_2026may.onnx"
_YUNET_2023 = "face_detection_yunet_2023mar.onnx"
_SFACE = "face_recognition_sface_2021dec.onnx"


@dataclass(frozen=True)
class _Model:
    filename: str
    urls: tuple[str, ...]
    sha256: str
    min_bytes: int


# Checksums are the files published by the OpenCV Zoo (Apache-2.0).
_YUNET_MODELS = (
    _Model(
        filename=_YUNET_2026,
        urls=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2026may.onnx",
        ),
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        min_bytes=150_000,
    ),
    _Model(
        filename=_YUNET_2023,
        urls=(
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        min_bytes=150_000,
    ),
)
_SFACE_MODEL = _Model(
    filename=_SFACE,
    urls=(
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    min_bytes=20_000_000,
)


def default_model_dir() -> Path:
    return Path.home() / ".noface" / "models"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(40)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


class OpenCVFaceEngine:
    """Local face engine. Models are downloaded once into ``model_dir``."""

    def __init__(
        self,
        model_dir: Path | None = None,
        det_score: float = 0.6,
        max_det_side: int = 1920,
        nms_threshold: float = 0.4,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise NofaceError(
                "This OpenCV build has no FaceDetectorYN or FaceRecognizerSF. "
                "Install opencv-python-headless >= 4.10."
            )
        _quiet_opencv_log()
        self.det_score = float(det_score)
        self.max_det_side = int(max_det_side)
        self.nms_threshold = float(nms_threshold)
        self.model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
        self._logged_sizes: set[tuple[int, int]] = set()
        recognizer_path = ensure_model(self.model_dir, _SFACE_MODEL)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self.detector = self._load_detector()

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        scales = detection_scales(width, height, self.max_det_side)
        key = (width, height)
        if key not in self._logged_sizes:
            self._logged_sizes.add(key)
            rendered = ", ".join(f"{scale:.3f}" for scale in scales)
            logger.debug("Detection scales for %dx%d: %s", width, height, rendered)

        found: list[Detection] = []
        for scale in scales:
            found.extend(self._detect_at(image, scale))
        return _nms(found, self.nms_threshold)

    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        if detection.raw is None:
            raise NofaceError("This face has no landmarks, so it cannot be embedded.")
        row = np.asarray(detection.raw, dtype=np.float32).reshape(-1)
        aligned = self.recognizer.alignCrop(image, row)
        feature = self.recognizer.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    def _load_detector(self) -> cv2.FaceDetectorYN:
        errors: list[str] = []
        for spec in _YUNET_MODELS:
            try:
                path = ensure_model(self.model_dir, spec)
                detector = cv2.FaceDetectorYN.create(
                    str(path),
                    "",
                    (320, 320),
                    self.det_score,
                    0.3,
                    5000,
                )
                detector.setInputSize((320, 240))
                detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
                logger.debug("Using face detector %s", spec.filename)
                return detector
            except Exception as exc:  # noqa: BLE001 - try the next published model
                errors.append(f"{spec.filename}: {exc}")
        detail = "\n".join(errors)
        raise NofaceError(f"Could not load a face detector.\n{detail}")

    def _detect_at(self, image: np.ndarray, scale: float) -> list[Detection]:
        height, width = image.shape[:2]
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        if min(resized_w, resized_h) < 48:
            return []
        if scale == 1.0 and resized_w == width and resized_h == height:
            view = image
        else:
            view = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        self.detector.setInputSize((resized_w, resized_h))
        faces = _unpack_faces(self.detector.detect(view))
        if faces is None:
            return []
        inv_x = width / resized_w
        inv_y = height / resized_h
        detections: list[Detection] = []
        for row in faces:
            mapped = np.asarray(row, dtype=np.float32).reshape(-1).copy()
            if mapped.shape[0] < 15:
                continue
            mapped[0:14:2] *= inv_x
            mapped[1:14:2] *= inv_y
            face_w = float(mapped[2])
            face_h = float(mapped[3])
            if face_w < 2 or face_h < 2:
                continue
            # Discard boxes that are still far outside the reliable band after scaling.
            apparent = min(face_w, face_h) * (resized_w / width)
            if apparent < MIN_DETECTOR_FACE * 0.4 or apparent > MAX_DETECTOR_FACE * 2.5:
                continue
            x = float(mapped[0])
            y = float(mapped[1])
            detections.append(
                Detection(
                    bbox=(x, y, x + face_w, y + face_h),
                    score=float(mapped[14]),
                    raw=mapped,
                )
            )
        return detections


def ensure_model(model_dir: Path, spec: _Model) -> Path:
    """Return a local model file, downloading and checking it when needed."""

    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / spec.filename
    if _matches(dest, spec):
        return dest
    if dest.exists():
        logger.warning("%s failed its checksum and will be downloaded again.", dest)
        dest.unlink()

    errors: list[str] = []
    for url in spec.urls:
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            logger.info("Downloading %s", spec.filename)
            _download(url, partial)
            if not _matches(partial, spec):
                raise NofaceError(
                    f"Downloaded {spec.filename} did not match the expected checksum."
                )
            partial.replace(dest)
            return dest
        except KeyboardInterrupt:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - fall through to the next mirror
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    tried = "\n".join(errors)
    raise NofaceError(
        f"Could not download {spec.filename}.\n{tried}\nPlace the file manually at {dest}."
    )


def _matches(path: Path, spec: _Model) -> bool:
    if not path.is_file() or path.stat().st_size < spec.min_bytes or is_lfs_pointer(path):
        return False
    return file_sha256(path) == spec.sha256


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "noface/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with dest.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name.removesuffix(".part"),
            leave=False,
        ) as bar:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bar.update(len(chunk))


def _unpack_faces(result: object) -> np.ndarray | None:
    # OpenCV 5 returns ``(count, None)`` when it finds nothing. The count is not useful.
    if result is None:
        return None
    faces = result[1] if isinstance(result, tuple) else result
    if faces is None:
        return None
    array = np.asarray(faces)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []
    for detection in ordered:
        if all(iou(detection.bbox, other.bbox) < iou_threshold for other in kept):
            kept.append(detection)
    return kept


def _quiet_opencv_log() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:  # noqa: BLE001 - logging setup differs across OpenCV builds
        pass
