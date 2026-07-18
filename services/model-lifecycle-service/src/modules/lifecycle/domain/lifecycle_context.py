from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    DETECTION = "detection"
    RECOGNIZER = "recognizer"


class LifecycleStage(str, Enum):
    LOOKUP = "lookup"
    DATASET = "dataset"
    TRAINING = "training"
    EXPORT = "export"
    TRACKING = "tracking"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class LifecycleContext:
    """Immutable, serialisable lineage carried by every lifecycle command."""

    lifecycle_id: str
    model_type: ModelType
    dataset_version: str
    raw_data_path: str
    stage: LifecycleStage = LifecycleStage.LOOKUP
    dataset_source: str | None = None
    dataset_root: str | None = None
    dataset_archive_path: str | None = None
    dataset_manifest_path: str | None = None
    training_data_path: str | None = None
    best_checkpoint_path: str | None = None
    last_checkpoint_path: str | None = None
    onnx_path: str | None = None
    mlflow_run_id: str | None = None
    mlflow_model_version: str | None = None

    def __post_init__(self) -> None:
        if not self.lifecycle_id.strip():
            raise ValueError("lifecycle_id is required")
        if not self.dataset_version.strip():
            raise ValueError("dataset_version is required")
        if not self.raw_data_path.strip():
            raise ValueError("raw_data_path is required")

    @property
    def stage_request_id(self) -> str:
        return f"{self.lifecycle_id}:{self.stage.value}"

    def advance(self, stage: LifecycleStage, **changes: str | None) -> "LifecycleContext":
        return replace(self, stage=stage, **changes)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in asdict(self).items()
        }
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LifecycleContext":
        return cls(
            lifecycle_id=_require(payload, "lifecycle_id"),
            model_type=ModelType(_require(payload, "model_type")),
            dataset_version=_require(payload, "dataset_version"),
            raw_data_path=_require(payload, "raw_data_path"),
            stage=LifecycleStage(payload.get("stage", LifecycleStage.LOOKUP.value)),
            dataset_source=_optional(payload, "dataset_source"),
            dataset_root=_optional(payload, "dataset_root"),
            dataset_archive_path=_optional(payload, "dataset_archive_path"),
            dataset_manifest_path=_optional(payload, "dataset_manifest_path"),
            training_data_path=_optional(payload, "training_data_path"),
            best_checkpoint_path=_optional(payload, "best_checkpoint_path"),
            last_checkpoint_path=_optional(payload, "last_checkpoint_path"),
            onnx_path=_optional(payload, "onnx_path"),
            mlflow_run_id=_optional(payload, "mlflow_run_id"),
            mlflow_model_version=_optional(payload, "mlflow_model_version"),
        )


def _require(payload: dict[str, Any], key: str) -> str:
    value = _optional(payload, key)
    if value is None:
        raise ValueError(f"lifecycle payload.{key} is required")
    return value


def _optional(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"lifecycle payload.{key} must be a non-empty string when present")
    return value.strip()
