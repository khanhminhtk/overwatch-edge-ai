from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import cv2


class VideoSourceType(str, Enum):
    CAMERA = "camera"
    STREAM_URL = "stream_url"
    IMAGE_PATH = "image_path"
    VIDEO_PATH = "video_path"


@dataclass(frozen=True)
class VideoSourceConfig:
    source_type: VideoSourceType
    camera_index: int
    stream_url: str
    image_path: str
    video_path: str
    width: int
    height: int
    fps: int
    params: dict[str, Any] = field(default_factory=dict)

    def configured_source(self) -> int | str:
        values: dict[VideoSourceType, int | str] = {
            VideoSourceType.CAMERA: self.camera_index,
            VideoSourceType.STREAM_URL: self.stream_url,
            VideoSourceType.IMAGE_PATH: self.image_path,
            VideoSourceType.VIDEO_PATH: self.video_path,
        }
        source = values[self.source_type]
        if isinstance(source, str) and not source.strip():
            raise ValueError(f"video.{self.source_type.value} must not be empty")
        return source


class VideoSource:
    def __init__(self, config: VideoSourceConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self, source: int | str | Path | None = None) -> None:
        """Open a camera index, stream URL, image path, or video path."""
        target: int | str
        if source is None:
            target = self._config.configured_source()
        elif isinstance(source, Path):
            target = str(source)
        else:
            target = source
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            self.release()
            raise RuntimeError(f"cannot open video source: {target}")
        if self._config.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        if self._config.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
