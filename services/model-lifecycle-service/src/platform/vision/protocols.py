from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VisionModelPort(ABC):
    @abstractmethod
    def execute(self, image_path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def execute_batch(self, images_paths: list[str]) -> list[Any]:
        raise NotImplementedError
