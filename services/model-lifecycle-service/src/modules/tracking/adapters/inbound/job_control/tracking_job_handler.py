from __future__ import annotations

import asyncio
from typing import Any

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.modules.tracking.application.use_case.workflow import MlflowWorkflow
from src.platform.logger import Logger


class TrackingJobHandler:
    def __init__(
        self,
        *,
        workflow: MlflowWorkflow,
        logger: Logger | None = None,
        pwd: str,
    ) -> None:
        self._workflow = workflow
        self._logger = logger or Logger("TrackingJobHandler")
        self._pwd = pwd

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        try:
            self._logger.info(f"Processing job {job.request_id} with event type {job.event_type}")
            await asyncio.to_thread(
                self._workflow.execute,
                checkpoint_names=self._resolve_checkpoint_names(job.payload),
                pwd=self._pwd,
            )
        except Exception as exc:
            self._logger.error(f"Error occurred while processing job {job.request_id}: {exc}")
            return JobResultDto(
                request_id=job.request_id,
                success=False,
                error_message=str(exc),
            )

        return JobResultDto(
            request_id=job.request_id,
            success=True,
        )

    def _resolve_checkpoint_names(self, payload: dict[str, Any] | None) -> list[str]:
        best_name = payload.get("checkpoint_best_name")
        last_name = payload.get("checkpoint_last_name")
        if isinstance(best_name, str) and best_name.strip() and isinstance(last_name, str) and last_name.strip():
            self._logger.info(f"Resolved checkpoint names: best_name={best_name}, last_name={last_name}")
            return [best_name, last_name]

        self._logger.warning("Checkpoint names are missing or invalid in the payload. Using default names.")
        return ["best.pt", "last.pt"]


class RecognizerTrackingJobHandler(TrackingJobHandler):
    def __init__(
        self,
        *,
        workflow: MlflowWorkflow,
        logger: Logger | None = None,
        pwd: str,
        default_checkpoint_names: list[str],
    ) -> None:
        super().__init__(workflow=workflow, logger=logger, pwd=pwd)
        self._default_checkpoint_names = default_checkpoint_names

    def _resolve_checkpoint_names(self, payload: dict[str, Any] | None) -> list[str]:
        checkpoint_names = super()._resolve_checkpoint_names(payload)
        if checkpoint_names == ["best.pt", "last.pt"] and self._default_checkpoint_names:
            self._logger.info(f"Using default checkpoint names: {self._default_checkpoint_names}")
            return self._default_checkpoint_names
        return checkpoint_names
