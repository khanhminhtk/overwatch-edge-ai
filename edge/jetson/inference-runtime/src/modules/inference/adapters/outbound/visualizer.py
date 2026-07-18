from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...application.state import InferenceState, inference_state
from ...domain.entities import Recognition


class VisualizerService:
    """Renders detector boxes and matching recognizer text onto the latest frame."""

    def __init__(self, state: InferenceState = inference_state) -> None:
        self._state = state

    def render_latest(self) -> np.ndarray | None:
        detection_snapshot = self._state.latest()
        recognition_snapshot = self._state.latest_recognitions()
        if detection_snapshot is None or recognition_snapshot is None:
            return None
        if recognition_snapshot.detection_sequence != detection_snapshot.sequence:
            return None

        return self.render(
            detection_snapshot.frame, list(recognition_snapshot.recognitions)
        )

    def render(self, frame: np.ndarray, recognitions: list[Recognition]) -> np.ndarray:
        frame = frame.copy()
        for recognition in recognitions:
            detection = recognition.detection
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 0),
                2,
            )
            label = _label(detection.confidence, recognition.text)
            frame = _draw_label(frame, label, detection.x1, detection.y1)
        return frame


def _label(confidence: float, text: str) -> str:
    return f"{text} {confidence:.2f}".strip()


def _draw_label(frame: np.ndarray, label: str, x: int, y: int) -> np.ndarray:
    """Draw Unicode text by converting the OpenCV BGR frame through Pillow."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_unicode_font(18)
    left, top, right, bottom = draw.textbbox((x + 4, y), label, font=font)
    label_top = max(0, y - (bottom - top) - 10)
    draw.rectangle((x, label_top, x + (right - left) + 8, y), fill=(0, 255, 0))
    draw.text((x + 4, label_top + 3), label, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    raise RuntimeError(
        "A Unicode TrueType font is required to render Vietnamese OCR text"
    )
