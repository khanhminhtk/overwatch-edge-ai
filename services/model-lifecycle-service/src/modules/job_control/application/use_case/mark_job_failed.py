from __future__ import annotations

from src.modules.job_control.application.ports.job_record_repository import (
    JobRecordRepository,
)


class MarkJobFailed:
    def __init__(self, *, repository: JobRecordRepository) -> None:
        self._repository = repository

    async def execute(self, *, request_id: str, error_message: str) -> bool:
        return await self._repository.mark_failed_by_request_id(
            request_id=request_id,
            error_message=error_message,
        )
