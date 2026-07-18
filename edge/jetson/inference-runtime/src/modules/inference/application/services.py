from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..domain.entities import Detection, Recognition
from ..domain.ports import DetectorModel
from .state import InferenceState, inference_state


@dataclass(frozen=True)
class DetectorSettings:
    input_width: int
    input_height: int
    confidence_threshold: float
    nms_threshold: float


class DetectorService:
    def __init__(
        self,
        model: DetectorModel,
        settings: DetectorSettings,
        state: InferenceState = inference_state,
    ) -> None:
        self._model = model
        self._settings = settings
        self._state = state

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_height, original_width = _validate_bgr_frame(frame)
        tensor = preprocess_bgr(
            frame, self._settings.input_width, self._settings.input_height
        )
        output = self._model.infer({"images": tensor}).get("output0")
        if output is None:
            raise RuntimeError("detector response is missing output0")
        detections = decode_yolo_output(
            output,
            original_width=original_width,
            original_height=original_height,
            input_width=self._settings.input_width,
            input_height=self._settings.input_height,
            confidence_threshold=self._settings.confidence_threshold,
            nms_threshold=self._settings.nms_threshold,
        )
        self._state.publish(frame, detections)
        return detections


def preprocess_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("detector input dimensions must be positive")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def decode_yolo_output(
    output: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
    confidence_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    values = np.asarray(output, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 5:
        raise ValueError("output0 must have shape [1, channels>=5, anchors]")
    if min(original_width, original_height, input_width, input_height) <= 0:
        raise ValueError("image dimensions must be positive")

    predictions = values[0]
    class_scores = predictions[4:]
    class_ids = class_scores.argmax(axis=0)
    confidences = class_scores.max(axis=0)
    candidates: list[Detection] = []
    for index in np.flatnonzero(confidences >= confidence_threshold):
        cx, cy, width, height = predictions[:4, index]
        x1 = max(0, round((cx - width / 2) * original_width / input_width))
        y1 = max(0, round((cy - height / 2) * original_height / input_height))
        x2 = min(original_width, round((cx + width / 2) * original_width / input_width))
        y2 = min(
            original_height, round((cy + height / 2) * original_height / input_height)
        )
        if x2 > x1 and y2 > y1:
            candidates.append(
                Detection(
                    int(class_ids[index]), float(confidences[index]), x1, y1, x2, y2
                )
            )
    return _non_maximum_suppression(candidates, nms_threshold)


def _non_maximum_suppression(
    candidates: list[Detection], threshold: float
) -> list[Detection]:
    selected: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(_iou(candidate, chosen) <= threshold for chosen in selected):
            selected.append(candidate)
    return selected


def _iou(left: Detection, right: Detection) -> float:
    intersection_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _validate_bgr_frame(frame: np.ndarray) -> tuple[int, int]:
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
        or not frame.size
    ):
        raise ValueError("detector expects a non-empty BGR uint8 HWC frame")
    return frame.shape[:2]


@dataclass(frozen=True)
class RecognizerSettings:
    num_patches: int
    patch_width: int
    patch_height: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    vocabulary: str
    blank_index: int
    num_classes: int


class RecognizerService:
    """Consumes the latest detector snapshot and reads each crop with CTC."""

    def __init__(
        self,
        model: DetectorModel,
        settings: RecognizerSettings,
        state: InferenceState = inference_state,
    ) -> None:
        self._model = model
        self._settings = settings
        self._state = state

    def recognize_latest(self) -> list[Recognition]:
        snapshot = self._state.latest()
        if snapshot is None:
            return []
        recognized = self._recognize(snapshot.frame, snapshot.detections)
        self._state.publish_recognitions(snapshot.sequence, recognized)
        return recognized

    def recognize_detections(self, detections: list[Detection]) -> list[Recognition]:
        """Recognize a selected subset of the latest detector boxes."""
        snapshot = self._state.latest()
        if snapshot is None:
            return []
        return self._recognize(snapshot.frame, tuple(detections))

    def recognize_frame(self, frame: np.ndarray) -> list[Recognition]:
        """Recognize a whole image when no detector crop is required."""
        height, width = _validate_bgr_frame(frame)
        full_frame = Detection(0, 1.0, 0, 0, width, height)
        return self._recognize(frame, (full_frame,))

    def _recognize(
        self, frame: np.ndarray, detections: tuple[Detection, ...]
    ) -> list[Recognition]:
        recognized: list[Recognition] = []
        for detection in detections:
            crop = frame[detection.y1 : detection.y2, detection.x1 : detection.x2]
            if not crop.size:
                continue
            tensor = preprocess_crop(crop, self._settings)
            logits = self._model.infer({"images": tensor}).get("logits")
            if logits is None:
                raise RuntimeError("recognizer response is missing logits")
            tokens, text = decode_ctc_logits(logits, self._settings)
            recognized.append(
                Recognition(
                    detection,
                    text,
                    tuple(tokens),
                    ctc_confidence(logits, self._settings.blank_index),
                )
            )
        return recognized


def preprocess_crop(crop: np.ndarray, settings: RecognizerSettings) -> np.ndarray:
    """Split a detection crop into horizontal RGB patches expected by recognizer.onnx."""
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or not crop.size:
        raise ValueError("recognizer expects a non-empty BGR uint8 HWC crop")
    if (
        settings.num_patches <= 0
        or settings.patch_width <= 0
        or settings.patch_height <= 0
    ):
        raise ValueError("recognizer patch dimensions must be positive")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if rgb.shape[1] < settings.num_patches:
        rgb = cv2.resize(rgb, (settings.num_patches, rgb.shape[0]))
    # Match RecognizerDataset: partition the original crop, then resize each patch.
    part_width = rgb.shape[1] // settings.num_patches
    patches = [
        cv2.resize(
            rgb[
                :,
                index * part_width : (index + 1) * part_width
                if index < settings.num_patches - 1
                else rgb.shape[1],
            ],
            (settings.patch_width, settings.patch_height),
        )
        for index in range(settings.num_patches)
    ]
    values = np.stack(patches).transpose(0, 3, 1, 2)[None].astype(np.float32) / 255.0
    mean = np.asarray(settings.mean, dtype=np.float32).reshape(1, 1, 3, 1, 1)
    std = np.asarray(settings.std, dtype=np.float32).reshape(1, 1, 3, 1, 1)
    if np.any(std <= 0):
        raise ValueError("recognizer normalization std must be positive")
    return np.ascontiguousarray((values - mean) / std)


def decode_ctc_logits(
    logits: np.ndarray, settings: RecognizerSettings
) -> tuple[list[int], str]:
    values = np.asarray(logits)
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or values.shape[2] != settings.num_classes
    ):
        raise ValueError(f"logits must have shape [1, time, {settings.num_classes}]")
    sequence = np.argmax(values[0], axis=-1).tolist()
    tokens: list[int] = []
    previous: int | None = None
    for index in sequence:
        index = int(index)
        if index != settings.blank_index and index != previous:
            tokens.append(index)
        previous = index
    return tokens, "".join(_token_to_text(index, settings) for index in tokens)


def ctc_confidence(logits: np.ndarray, blank_index: int) -> float:
    """Mean softmax probability for non-blank CTC steps after collapse."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError("logits must have shape [1, time, classes]")
    shifted = values[0] - values[0].max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    sequence = np.argmax(values[0], axis=-1)
    selected: list[float] = []
    previous: int | None = None
    for timestep, index in enumerate(sequence.tolist()):
        if index != blank_index and index != previous:
            selected.append(float(probabilities[timestep, index]))
        previous = index
    return float(np.mean(selected)) if selected else 0.0


def _token_to_text(index: int, settings: RecognizerSettings) -> str:
    if 1 <= index <= len(settings.vocabulary):
        return settings.vocabulary[index - 1]
    return "<unk>"
