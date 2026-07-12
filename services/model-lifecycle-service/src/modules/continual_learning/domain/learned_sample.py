from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ProcessedResult:
    source_image: str
    detection_dir: str
    recognizer_dir: str
    detection_samples: int
    recognizer_samples: int
