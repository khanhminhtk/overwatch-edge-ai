from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.modules.lifecycle.domain import LifecycleContext, LifecycleStage


@dataclass(frozen=True, slots=True)
class LifecycleCommand:
    request_id: str
    event_type: str
    payload: dict[str, Any]


class LifecycleWorkflow:
    """Pure transition rules; Kafka/Postgres adapters only persist/publish its commands."""

    def start(self, context: LifecycleContext) -> LifecycleCommand:
        return self._command(context, "dataset_lookup_requested")

    def after_dataset(self, context: LifecycleContext, result: dict[str, Any]) -> LifecycleCommand:
        updated = context.advance(
            LifecycleStage.DATASET,
            dataset_root=_string(result, "dataset_root"),
            dataset_archive_path=_string(result, "archive_path"),
            dataset_manifest_path=_string(result, "manifest_path"),
            training_data_path=_string(result, "training_data_path", required=False),
        )
        return self._command(updated.advance(LifecycleStage.TRAINING), self._training_event(updated))

    def after_training(self, context: LifecycleContext, result: dict[str, Any]) -> LifecycleCommand:
        updated = context.advance(
            LifecycleStage.EXPORT,
            best_checkpoint_path=_string(result, "best_checkpoint_path", required=False),
            last_checkpoint_path=_string(result, "last_checkpoint_path", required=False),
        )
        return self._command(updated, self._export_event(updated))

    def after_export(self, context: LifecycleContext, result: dict[str, Any]) -> LifecycleCommand:
        updated = context.advance(LifecycleStage.TRACKING, onnx_path=_string(result, "onnx_path"))
        return self._command(updated, self._tracking_event(updated))

    def complete(self, context: LifecycleContext, result: dict[str, Any]) -> LifecycleContext:
        return context.advance(
            LifecycleStage.COMPLETED,
            mlflow_run_id=_string(result, "mlflow_run_id", required=False),
            mlflow_model_version=_string(result, "mlflow_model_version", required=False),
        )

    @staticmethod
    def _command(context: LifecycleContext, event_type: str) -> LifecycleCommand:
        return LifecycleCommand(context.stage_request_id, event_type, context.to_payload())

    @staticmethod
    def _training_event(context: LifecycleContext) -> str:
        return f"train_requested_{'detection' if context.model_type.value == 'detection' else 'recognizer'}"

    @staticmethod
    def _export_event(context: LifecycleContext) -> str:
        return context.model_type.value

    @staticmethod
    def _tracking_event(context: LifecycleContext) -> str:
        return context.model_type.value


def _string(result: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    value = result.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"stage result.{key} is required")
    return value.strip()
