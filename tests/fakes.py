"""Face engines whose detections are scripted by the test."""

from __future__ import annotations

import numpy as np

from noface.types import Detection


class ScriptedEngine:
    """Each ``detect`` call returns the next scripted frame.

    A frame is a list of ``(detection, embedding)`` pairs. Embeddings are
    looked up by detection object, so two faces can share a box across frames
    and still receive different features.
    """

    def __init__(self, script: list[list[tuple[Detection, np.ndarray]]]) -> None:
        self.script = script
        self.calls = 0
        self._features: dict[int, np.ndarray] = {}

    def detect(self, image: np.ndarray) -> list[Detection]:
        pairs = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        self._features = {id(detection): feature for detection, feature in pairs}
        return [detection for detection, _feature in pairs]

    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        return self._features[id(detection)]
