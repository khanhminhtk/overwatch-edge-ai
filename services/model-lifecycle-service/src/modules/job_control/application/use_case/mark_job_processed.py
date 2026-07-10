from __future__ import annotations

from src.modules.job_control.application.ports.job_record_repository import (
    JobRecordRepository,
)


class MarkJobProcessed:
    def __init__(self, *, repository: JobRecordRepository) -> None:
        self._repository = repository

    async def execute(self, *, request_id: str) -> bool:
        return await self._repository.mark_processed_by_request_id(request_id)
