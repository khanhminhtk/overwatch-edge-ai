from __future__ import annotations

import asyncio

from src.modules.dataset.application.use_case import (
    CreateGodDatasetDetection,
    CreateGodDatasetRecognizer,
)
from src.modules.lifecycle.application import DatasetWorkspaceMaterializer
from src.modules.lifecycle.domain import LifecycleContext, ModelType
from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.platform.logger import Logger


class DatasetJobHandler:
    def __init__(
        self,
        *,
        detection_dataset: CreateGodDatasetDetection,
        recognizer_dataset: CreateGodDatasetRecognizer,
        detection_event_type: str,
        recognizer_event_type: str,
        workspace_materializer: DatasetWorkspaceMaterializer | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._detection_dataset = detection_dataset
        self._recognizer_dataset = recognizer_dataset
        self._detection_event_type = detection_event_type
        self._recognizer_event_type = recognizer_event_type
        self._workspace_materializer = workspace_materializer
        self._logger = logger or Logger("DatasetJobHandler")

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        payload = job.payload or {}
        source_data_path = payload.get("source_data_path", "")
        subset_percent = float(payload.get("subset_percent", 10.0))
        output_root = payload.get("output_root")
        dataset_version = payload.get("dataset_version", "v1")

        if not source_data_path:
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message="Missing 'source_data_path' in job payload",
            )

        try:
            self._logger.info(
                "[DATASET_JOB_STARTED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
                f"source_data_path={source_data_path}",
                f"subset_percent={subset_percent}",
            )

            if job.event_type == self._detection_event_type:
                model_type = ModelType.DETECTION
                dataset_result = await asyncio.to_thread(
                    self._detection_dataset.execute,
                    source_data_path=source_data_path,
                    subset_percent=subset_percent,
                    output_root=output_root,
                    dataset_version=dataset_version,
                )
            elif job.event_type == self._recognizer_event_type:
                model_type = ModelType.RECOGNIZER
                dataset_result = await asyncio.to_thread(
                    self._recognizer_dataset.execute,
                    source_data_path=source_data_path,
                    subset_percent=subset_percent,
                    output_root=output_root,
                    dataset_version=dataset_version,
                )
            else:
                return JobResultDto(
                    request_id=job.request_id,
                    success=False,
                    error_message=f"Unsupported dataset event_type='{job.event_type}'",
                )
        except Exception as exc:
            self._logger.exception(
                "[DATASET_JOB_FAILED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )

        self._logger.info(
            "[DATASET_JOB_SUCCEEDED]",
            f"request_id={job.request_id}",
            f"event_type={job.event_type}",
        )
        result = {
            "dataset_root": dataset_result.dataset_root,
            "archive_path": dataset_result.archive_path,
            "manifest_path": dataset_result.manifest_path,
            "selected_samples": dataset_result.selected_samples,
        }
        lifecycle_id = payload.get("lifecycle_id")
        if self._workspace_materializer is not None and lifecycle_id is not None:
            if not isinstance(lifecycle_id, str) or not lifecycle_id.strip():
                return JobResultDto(
                    request_id=job.request_id,
                    success=False,
                    error_message="lifecycle_id must be a non-empty string when provided",
                )
            context = LifecycleContext(
                lifecycle_id=lifecycle_id,
                model_type=model_type,
                dataset_version=dataset_version,
                raw_data_path=source_data_path,
                dataset_root=dataset_result.dataset_root,
                dataset_archive_path=dataset_result.archive_path,
                dataset_manifest_path=dataset_result.manifest_path,
            )
            materialized = await asyncio.to_thread(self._workspace_materializer.materialize, context)
            result["training_data_path"] = materialized.training_data_path

        return JobResultDto(
            request_id=job.request_id,
            success=True,
            result=result,
        )
