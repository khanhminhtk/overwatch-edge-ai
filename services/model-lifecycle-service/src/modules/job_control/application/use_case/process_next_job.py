from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.use_case.claim_next_pending_job import (
    ClaimNextPendingJob,
)
from src.modules.job_control.application.use_case.mark_job_failed import (
    MarkJobFailed,
)
from src.modules.job_control.application.use_case.mark_job_processed import (
    MarkJobProcessed,
)
from src.modules.job_control.application.ports.job_handler import JobHandler
from src.modules.job_control.adapters.outbound.messaging.kafka.job_success_event_publisher import (
    JobSuccessEventPublisher,
)
from src.platform.logger import Logger


class ProcessNextJob:
    _PERSIST_RETRY_ATTEMPTS = 3
    _PERSIST_RETRY_DELAY_SECONDS = 1.0

    def __init__(
        self,
        *,
        claim_job: ClaimNextPendingJob,
        handler: JobHandler,
        mark_processed: MarkJobProcessed,
        mark_failed: MarkJobFailed,
        logger: Logger,
        job_name: str = "unknown",
        success_event_publisher: JobSuccessEventPublisher | None = None,
    ) -> None:
        self._claim_job = claim_job
        self._handler = handler
        self._mark_processed = mark_processed
        self._mark_failed = mark_failed
        self._logger = logger
        self._job_name = job_name
        self._success_event_publisher = success_event_publisher
        self._inflight_job: ClaimedJobDto | None = None

    async def execute(self, *, server_id: str) -> JobResultDto | None:
        claimed_job = await self._claim_job.execute(server_id=server_id)
        if claimed_job is None:
            self._logger.info("[PROCESS_NEXT_JOB_EMPTY]", f"server_id={server_id}")
            return None

        self._logger.info(
            "[PROCESS_NEXT_JOB_CLAIMED]",
            f"server_id={server_id}",
            f"request_id={claimed_job.request_id}",
            f"event_type={claimed_job.event_type}",
        )

        self._inflight_job = claimed_job
        started_at = time.perf_counter()
        finalized = False
        try:
            try:
                result = await self._handler.handle(claimed_job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception(
                    "[PROCESS_NEXT_JOB_HANDLER_ERROR]",
                    f"server_id={server_id}",
                    f"request_id={claimed_job.request_id}",
                    f"event_type={claimed_job.event_type}",
                )
                result = JobResultDto(
                    request_id=claimed_job.request_id,
                    success=False,
                    error_message=str(exc),
                )

            duration_ms = (time.perf_counter() - started_at) * 1000
            self._logger.info(
                "[PROCESS_NEXT_JOB_HANDLER_COMPLETED]",
                f"server_id={server_id}",
                f"request_id={claimed_job.request_id}",
                f"success={result.success}",
                f"duration_ms={duration_ms:.2f}",
            )
            if result.success:
                await self._persist_job_state(
                    server_id=server_id,
                    request_id=result.request_id,
                    target_status="PROCESSED",
                    persist_action=lambda: self._mark_processed.execute(
                        request_id=result.request_id
                    ),
                )
                finalized = True
                self._publish_success_event(
                    server_id=server_id,
                    claimed_job=claimed_job,
                    result=result.result,
                )
                self._logger.info(
                    "[PROCESS_NEXT_JOB_SUCCEEDED]",
                    f"server_id={server_id}",
                    f"request_id={result.request_id}",
                )
                return result

            await self._persist_job_state(
                server_id=server_id,
                request_id=result.request_id,
                target_status="FAILED",
                persist_action=lambda: self._mark_failed.execute(
                    request_id=result.request_id,
                    error_message=result.error_message or "unknown job error",
                ),
            )
            finalized = True
            self._logger.error(
                "[PROCESS_NEXT_JOB_FAILED]",
                f"server_id={server_id}",
                f"request_id={result.request_id}",
                f"error={result.error_message or 'unknown job error'}",
            )
            return result
        finally:
            if finalized and self._inflight_job is not None:
                if self._inflight_job.request_id == claimed_job.request_id:
                    self._inflight_job = None

    async def fail_inflight_job(
        self,
        *,
        server_id: str,
        reason: str,
    ) -> None:
        if self._inflight_job is None:
            return
        request_id = self._inflight_job.request_id
        await self._persist_job_state(
            server_id=server_id,
            request_id=request_id,
            target_status="FAILED",
            persist_action=lambda: self._mark_failed.execute(
                request_id=request_id,
                error_message=reason,
            ),
        )
        self._logger.warning(
            "[PROCESS_NEXT_JOB_INFLIGHT_MARKED_FAILED]",
            f"server_id={server_id}",
            f"request_id={request_id}",
            f"reason={reason}",
        )
        self._inflight_job = None

    async def _persist_job_state(
        self,
        *,
        server_id: str,
        request_id: str,
        target_status: str,
        persist_action: Callable[[], Awaitable[bool]],
    ) -> None:
        for attempt in range(1, self._PERSIST_RETRY_ATTEMPTS + 1):
            try:
                await persist_action()
                self._logger.info(
                    "[PROCESS_NEXT_JOB_STATE_PERSISTED]",
                    f"server_id={server_id}",
                    f"request_id={request_id}",
                    f"target_status={target_status}",
                    f"attempt={attempt}",
                )
                return
            except Exception as exc:
                if attempt >= self._PERSIST_RETRY_ATTEMPTS:
                    self._logger.exception(
                        "[PROCESS_NEXT_JOB_STATE_PERSIST_FAILED]",
                        f"server_id={server_id}",
                        f"request_id={request_id}",
                        f"target_status={target_status}",
                        f"attempts={attempt}",
                    )
                    raise
                self._logger.warning(
                    "[PROCESS_NEXT_JOB_STATE_PERSIST_RETRY]",
                    f"server_id={server_id}",
                    f"request_id={request_id}",
                    f"target_status={target_status}",
                    f"attempt={attempt}",
                    f"error={exc}",
                )
                await asyncio.sleep(self._PERSIST_RETRY_DELAY_SECONDS)

    def _publish_success_event(
        self,
        *,
        server_id: str,
        claimed_job: ClaimedJobDto,
        result: dict[str, object] | None,
    ) -> None:
        if self._success_event_publisher is None:
            return
        try:
            self._success_event_publisher.publish(
                job_name=self._job_name,
                server_id=server_id,
                claimed_job=claimed_job,
                result=result,
            )
        except Exception:
            self._logger.exception(
                "[JOB_SUCCESS_EVENT_PUBLISH_FAILED]",
                f"server_id={server_id}",
                f"request_id={claimed_job.request_id}",
                f"job_name={self._job_name}",
            )
