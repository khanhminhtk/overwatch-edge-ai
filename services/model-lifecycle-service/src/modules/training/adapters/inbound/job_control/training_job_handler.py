from __future__ import annotations

import asyncio
from typing import Any

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.modules.training.application.use_case import (
    TrainDetectionUseCase,
    TrainRecognizerUseCase,
)
from src.modules.training.domain.value_objects.training_spec import TrainingSpec
from src.platform.logger import Logger


class TrainingJobHandler:
    def __init__(
        self,
        *,
        recognizer_training: TrainRecognizerUseCase,
        detection_training: TrainDetectionUseCase,
        recognizer_event_type: str,
        detection_event_type: str,
        default_mode: str = "local",
        logger: Logger | None = None,
    ) -> None:
        self._recognizer_training = recognizer_training
        self._detection_training = detection_training
        self._recognizer_event_type = recognizer_event_type
        self._detection_event_type = detection_event_type
        self._default_mode = default_mode
        self._logger = logger or Logger("TrainingJobHandler")

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        payload = job.payload or {}
        mode = payload.get("mode") or self._default_mode
        dataset_version = payload.get("dataset_version", "")
        train_config_uri = payload.get("train_config_uri")

        if job.event_type == self._recognizer_event_type:
            model_name = "recognizer"
            use_case = self._recognizer_training
        elif job.event_type == self._detection_event_type:
            model_name = "detector"
            use_case = self._detection_training
        else:
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=f"Unsupported training event_type='{job.event_type}'",
            )

        spec = TrainingSpec(
            model_name=model_name,
            dataset_version=dataset_version,
            train_config_uri=train_config_uri,
            mode=mode,
        )

        try:
            self._logger.info(
                "[TRAINING_JOB_STARTED]",
                f"request_id={job.request_id}",
                f"model_name={model_name}",
                f"event_type={job.event_type}",
                f"mode={mode}",
                f"dataset_version={dataset_version}",
            )
            await asyncio.to_thread(use_case.execute, spec)
        except Exception as exc:
            self._logger.exception(
                "[TRAINING_JOB_FAILED]",
                f"request_id={job.request_id}",
                f"model_name={model_name}",
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )

        self._logger.info(
            "[TRAINING_JOB_SUCCEEDED]",
            f"request_id={job.request_id}",
            f"model_name={model_name}",
        )
        return JobResultDto(
            request_id=job.request_id,
            success=True,
        )
