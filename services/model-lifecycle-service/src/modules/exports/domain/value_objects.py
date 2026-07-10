from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExportSpec:
    model_type: str
    checkpoint_path: Path
    output_path: Path
    exporter: str = "legacy"
    training_config_path: Path | None = None
    training_env_path: Path | None = None
    project_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_path: Path
    success: bool
    duration_ms: float | None = None
