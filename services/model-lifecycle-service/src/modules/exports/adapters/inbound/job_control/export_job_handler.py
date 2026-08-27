from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.modules.exports.application.use_case import (
    ExportDetectionUseCase,
    ExportRecognizerUseCase,
)
from src.modules.exports.domain.value_objects import ExportSpec
from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.modules.tracking.domain.value_objects import (
    DetectionConfig,
    RecognizerConfig,
)
from src.platform.logger import Logger


@dataclass(frozen=True, slots=True)
class _ExportTarget:
    model_type: str
    output_name: str
    checkpoint_dir: str
    default_checkpoint_name: str
    training_config_path: str
    training_env_path: str


class ExportJobHandler:
    def __init__(
        self,
        *,
        detection_export: ExportDetectionUseCase,
        recognizer_export: ExportRecognizerUseCase,
        detection_config: DetectionConfig,
        recognizer_config: RecognizerConfig,
        detection_event_type: str,
        recognizer_event_type: str,
        project_root: Path,
        onnx_export_dir: str = "artifacts/onnx",
        logger: Logger | None = None,
    ) -> None:
        self._detection_export = detection_export
        self._recognizer_export = recognizer_export
        self._detection_event_type = detection_event_type
        self._recognizer_event_type = recognizer_event_type
        self._project_root = project_root
        self._onnx_export_dir = Path(onnx_export_dir)
        self._logger = logger or Logger("ExportJobHandler")
        self._targets = {
            detection_event_type: _ExportTarget(
                model_type="detection",
                output_name="detection.onnx",
                checkpoint_dir=detection_config.checkpoint_dir,
                default_checkpoint_name=detection_config.best_checkpoint_name,
                training_config_path=detection_config.training_config_path,
                training_env_path=detection_config.training_env_path,
            ),
            recognizer_event_type: _ExportTarget(
                model_type="recognizer",
                output_name="recognizer.onnx",
                checkpoint_dir=recognizer_config.checkpoint_dir,
                default_checkpoint_name=recognizer_config.best_checkpoint_name,
                training_config_path=recognizer_config.training_config_path,
                training_env_path=recognizer_config.training_env_path,
            ),
        }

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        try:
            target = self._targets[job.event_type]
        except KeyError:
            error_message = f"Unsupported export event_type={job.event_type}"
            self._logger.error("[EXPORT_JOB_UNSUPPORTED_EVENT_TYPE]", error_message)
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=error_message,
            )

        try:
            spec = self._build_spec(job=job, target=target)
            self._logger.info(
                "[EXPORT_JOB_STARTED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
                f"checkpoint={spec.checkpoint_path}",
                f"output={spec.output_path}",
            )
            await asyncio.to_thread(self._execute_export, job.event_type, spec)
        except Exception as exc:
            self._logger.exception(
                "[EXPORT_JOB_FAILED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )

        self._logger.info(
            "[EXPORT_JOB_SUCCEEDED]",
            f"request_id={job.request_id}",
            f"event_type={job.event_type}",
        )
        return JobResultDto(
            request_id=job.request_id,
            success=True,
        )

    def _execute_export(self, event_type: str, spec: ExportSpec) -> None:
        if event_type == self._detection_event_type:
            self._detection_export.execute(spec)
            return
        self._recognizer_export.execute(spec)

    def _build_spec(self, *, job: ClaimedJobDto, target: _ExportTarget) -> ExportSpec:
        payload = job.payload or {}
        checkpoint_name = payload.get("checkpoint_best_name")
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            checkpoint_name = target.default_checkpoint_name

        checkpoint_path = self._resolve_path(Path(target.checkpoint_dir) / checkpoint_name)
        if checkpoint_name != target.default_checkpoint_name and not checkpoint_path.exists():
            fallback_checkpoint_path = self._resolve_path(
                Path(target.checkpoint_dir) / target.default_checkpoint_name
            )
            self._logger.warning(
                "[EXPORT_JOB_CHECKPOINT_FALLBACK]",
                f"requested_checkpoint={checkpoint_path}",
                f"fallback_checkpoint={fallback_checkpoint_path}",
            )
            checkpoint_path = fallback_checkpoint_path

        output_path = self._resolve_path(self._onnx_export_dir) / target.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return ExportSpec(
            model_type=target.model_type,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            training_config_path=self._resolve_path(Path(target.training_config_path)),
            training_env_path=self._resolve_path(Path(target.training_env_path)),
            project_root=self._project_root,
        )

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._project_root / path
