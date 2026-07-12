from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DetectionSampleFeature:
    image_path: Path
    label_path: Path
    pixel_count: int
    sum_rgb: tuple[float, float, float]
    sum_sq_rgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _RecognitionSampleRef:
    group_name: str
    image_name: str
    image_path: Path
    label: str


@dataclass(frozen=True, slots=True)
class RecognizerSampleFeature:
    sample: _RecognitionSampleRef
    pixel_count: int
    sum_rgb: tuple[float, float, float]
    sum_sq_rgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GodDatasetResult:
    dataset_root: str
    archive_path: str
    manifest_path: str
    selected_samples: int
