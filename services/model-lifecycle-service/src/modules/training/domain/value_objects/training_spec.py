from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    model_name: str
    dataset_version: str
    train_config_uri: str | None = None
    mode: str = "local"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    success: bool
    duration_ms: float | None = None
