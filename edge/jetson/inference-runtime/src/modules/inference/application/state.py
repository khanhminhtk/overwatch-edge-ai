from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from ..domain.entities import Detection, Recognition


@dataclass(frozen=True)
class DetectionSnapshot:
    sequence: int
    frame: np.ndarray
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class RecognitionSnapshot:
    detection_sequence: int
    recognitions: tuple[Recognition, ...]


class InferenceState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sequence = 0
        self._latest: DetectionSnapshot | None = None
        self._latest_recognitions: RecognitionSnapshot | None = None

    def publish(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> DetectionSnapshot:
        frame_copy = np.ascontiguousarray(frame).copy()
        frame_copy.setflags(write=False)
        with self._lock:
            self._sequence += 1
            self._latest = DetectionSnapshot(
                self._sequence, frame_copy, tuple(detections)
            )
            return self._latest

    def latest(self) -> DetectionSnapshot | None:
        with self._lock:
            return self._latest

    def publish_recognitions(
        self,
        detection_sequence: int,
        recognitions: list[Recognition],
    ) -> RecognitionSnapshot:
        with self._lock:
            self._latest_recognitions = RecognitionSnapshot(
                detection_sequence, tuple(recognitions)
            )
            return self._latest_recognitions

    def latest_recognitions(self) -> RecognitionSnapshot | None:
        with self._lock:
            return self._latest_recognitions

inference_state = InferenceState()
