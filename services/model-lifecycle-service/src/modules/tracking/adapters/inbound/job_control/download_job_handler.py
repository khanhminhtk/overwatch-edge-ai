from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.platform.logger import Logger


class ModelDownloadUseCase(Protocol):
    def download_champion_to_file(self, artifact_path: str, output_path: str) -> str: ...

    def download_version_to_file(
        self,
        version: str,
        artifact_path: str,
        output_path: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _DownloadTarget:
    default_checkpoint_name: str
    use_case: ModelDownloadUseCase


class DownloadJobHandler:
    def __init__(
        self,
        *,
        detection_download: ModelDownloadUseCase,
        recognizer_download: ModelDownloadUseCase,
        detection_event_type: str,
        recognizer_event_type: str,
        detection_default_checkpoint_name: str,
        recognizer_default_checkpoint_name: str,
        pwd: str,
        download_timeout_seconds: float = 120.0,
        logger: Logger | None = None,
    ) -> None:
        self._pwd = Path(pwd)
        self._download_timeout_seconds = download_timeout_seconds
        self._logger = logger or Logger("DownloadJobHandler")
        self._targets = {
            detection_event_type: _DownloadTarget(
                default_checkpoint_name=detection_default_checkpoint_name,
                use_case=detection_download,
            ),
            recognizer_event_type: _DownloadTarget(
                default_checkpoint_name=recognizer_default_checkpoint_name,
                use_case=recognizer_download,
            ),
        }

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        try:
            target = self._targets[job.event_type]
        except KeyError:
            error_message = f"Unsupported download event_type={job.event_type}"
            self._logger.error("[DOWNLOAD_JOB_UNSUPPORTED_EVENT_TYPE]", error_message)
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=error_message,
            )

        payload = job.payload or {}
        checkpoint_name = payload.get("checkpoint_best_name")
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            checkpoint_name = target.default_checkpoint_name

        artifact_path = f"checkpoints/{checkpoint_name}"
        try:
            output_path = self._resolve_output_path(payload=payload)
        except ValueError as exc:
            self._logger.error(
                "[DOWNLOAD_JOB_INVALID_PAYLOAD]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
                str(exc),
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        version = payload.get("model_version")
        try:
            self._logger.info(
                "[DOWNLOAD_JOB_STARTED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
                f"artifact_path={artifact_path}",
                f"output_path={output_path}",
                f"model_version={version}",
                f"timeout_seconds={self._download_timeout_seconds}",
            )
            if isinstance(version, str) and version.strip():
                await asyncio.wait_for(
                    asyncio.to_thread(
                        target.use_case.download_version_to_file,
                        version.strip(),
                        artifact_path,
                        str(output_path),
                    ),
                    timeout=self._download_timeout_seconds,
                )
            else:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        target.use_case.download_champion_to_file,
                        artifact_path,
                        str(output_path),
                    ),
                    timeout=self._download_timeout_seconds,
                )
        except asyncio.TimeoutError:
            error_message = (
                "Download timed out after "
                f"{self._download_timeout_seconds:.1f}s for event_type={job.event_type}"
            )
            self._logger.error(
                "[DOWNLOAD_JOB_TIMEOUT]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
                f"artifact_path={artifact_path}",
                f"output_path={output_path}",
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=error_message,
            )
        except Exception as exc:
            self._logger.exception(
                "[DOWNLOAD_JOB_FAILED]",
                f"request_id={job.request_id}",
                f"event_type={job.event_type}",
            )
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )

        self._logger.info(
            "[DOWNLOAD_JOB_SUCCEEDED]",
            f"request_id={job.request_id}",
            f"event_type={job.event_type}",
            f"output_path={output_path}",
        )
        return JobResultDto(
            request_id=job.request_id,
            success=True,
        )

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._pwd / path

    def _resolve_output_path(
        self,
        *,
        payload: dict[str, object],
    ) -> Path:
        output_path_value = payload.get("output_path")
        if isinstance(output_path_value, str) and output_path_value.strip():
            return self._resolve_path(Path(output_path_value.strip()))
        raise ValueError("Missing required payload.output_path for download job")
