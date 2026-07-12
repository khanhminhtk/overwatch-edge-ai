from __future__ import annotations

import asyncio

from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.modules.job_control.application.use_case.process_next_job import ProcessNextJob
from src.platform.logger import Logger


class PollingJobRunner:
    def __init__(
        self,
        *,
        process_next_job: ProcessNextJob,
        server_id: str,
        event_type: str,
        logger: Logger,
        idle_sleep_seconds: float = 2.0,
    ) -> None:
        self._process_next_job = process_next_job
        self._server_id = server_id
        self._event_type = event_type
        self._logger = logger
        self._idle_sleep_seconds = idle_sleep_seconds
        self._stop_requested = False

    async def run_once(self) -> JobResultDto | None:
        self._logger.info(
            "[JOB_POLLING_TICK]",
            f"server_id={self._server_id}",
            f"event_type={self._event_type}",
        )
        result = await self._process_next_job.execute(server_id=self._server_id)
        if result is None:
            self._logger.info(
                "[JOB_POLLING_IDLE]",
                f"server_id={self._server_id}",
                f"event_type={self._event_type}",
                f"sleep_seconds={self._idle_sleep_seconds}",
            )
            return None

        self._logger.info(
            "[JOB_POLLING_RESULT]",
            f"server_id={self._server_id}",
            f"event_type={self._event_type}",
            f"request_id={result.request_id}",
            f"success={result.success}",
        )
        return result

    async def run_forever(self) -> None:
        self._logger.info(
            "[JOB_POLLING_RUNNER_STARTED]",
            f"server_id={self._server_id}",
            f"event_type={self._event_type}",
        )
        while not self._stop_requested:
            try:
                result = await self.run_once()
                if result is None:
                    await asyncio.sleep(self._idle_sleep_seconds)
            except Exception:
                self._logger.exception(
                    "[JOB_POLLING_ERROR]",
                    f"server_id={self._server_id}",
                    f"event_type={self._event_type}",
                )
                await asyncio.sleep(self._idle_sleep_seconds)

    def stop(self) -> None:
        self._stop_requested = True
        self._logger.info(
            "[JOB_POLLING_RUNNER_STOPPED]",
            f"server_id={self._server_id}",
            f"event_type={self._event_type}",
        )

    async def shutdown(self, *, reason: str) -> None:
        self._stop_requested = True
        self._logger.info(
            "[JOB_POLLING_RUNNER_SHUTDOWN_REQUESTED]",
            f"server_id={self._server_id}",
            f"event_type={self._event_type}",
            f"reason={reason}",
        )
        try:
            await self._process_next_job.fail_inflight_job(
                server_id=self._server_id,
                reason=reason,
            )
        except Exception:
            self._logger.exception(
                "[JOB_POLLING_RUNNER_SHUTDOWN_FAILED]",
                f"server_id={self._server_id}",
                f"event_type={self._event_type}",
                f"reason={reason}",
            )
