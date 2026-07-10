from __future__ import annotations

from dataclasses import dataclass

from src.modules.tracking.domain.value_objects.recognizer_config import RecognizerConfig


@dataclass
class RecognizerRun:
    experiment_name: str
    run_name: str
    config: RecognizerConfig
    run_id: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecognizerRun):
            return NotImplemented
        return self.run_id == other.run_id

    def __hash__(self) -> int:
        return hash(self.run_id)
