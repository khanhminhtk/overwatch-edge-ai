from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
from pydantic.dataclasses import dataclass as validated_dataclass


@validated_dataclass(frozen=True)
class LowQualityCaptureConfig:
    enabled: bool = True
    success_output_dir: str = "data/success"
    fail_output_dir: str = "data/fail"
    success_min_detection_confidence: float = 0.50
    success_min_ocr_confidence: float = 0.85
    fail_max_detection_confidence: float = 0.35
    fail_max_ocr_confidence: float = 0.70
    min_frame_difference: float = 0.12
    min_save_interval_seconds: float = 5.0


class LowQualityFrameStore:
    """Writes original frames into success/fail folders without near-duplicates."""

    def __init__(self, config: LowQualityCaptureConfig, service_root: Path) -> None:
        _validate_config(config)
        self._config = config
        self._output_dirs = {
            "success": _resolve_dir(service_root, config.success_output_dir),
            "fail": _resolve_dir(service_root, config.fail_output_dir),
        }
        self._last_thumbnails: dict[str, np.ndarray] = {}
        self._last_saved_at: dict[str, float] = {}

    @property
    def config(self) -> LowQualityCaptureConfig:
        return self._config

    def save_if_distinct(self, category: str, frame: np.ndarray) -> Path | None:
        if not self._config.enabled:
            return None
        if category not in self._output_dirs:
            raise ValueError(f"unsupported capture category: {category}")
        now = time.monotonic()
        if now - self._last_saved_at.get(category, -float("inf")) < self._config.min_save_interval_seconds:
            return None
        thumbnail = _thumbnail(frame)
        previous = self._last_thumbnails.get(category)
        if (
            previous is not None
            and _frame_difference(thumbnail, previous) < self._config.min_frame_difference
        ):
            return None
        output_dir = self._output_dirs[category]
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime(f"{category}_%Y%m%dT%H%M%S_%fZ.jpg")
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"cannot save {category} OCR frame: {output_path}")
        self._last_thumbnails[category] = thumbnail
        self._last_saved_at[category] = now
        return output_path


def _validate_config(config: LowQualityCaptureConfig) -> None:
    for name, value in (
        ("success_min_detection_confidence", config.success_min_detection_confidence),
        ("success_min_ocr_confidence", config.success_min_ocr_confidence),
        ("fail_max_detection_confidence", config.fail_max_detection_confidence),
        ("fail_max_ocr_confidence", config.fail_max_ocr_confidence),
        ("min_frame_difference", config.min_frame_difference),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"capture.{name} must be between 0 and 1")
    if config.min_save_interval_seconds < 0:
        raise ValueError("capture.min_save_interval_seconds must be non-negative")


def _resolve_dir(service_root: Path, configured_dir: str) -> Path:
    path = Path(configured_dir)
    return path if path.is_absolute() else service_root / path


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)


def _frame_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(left, right))) / 255.0
