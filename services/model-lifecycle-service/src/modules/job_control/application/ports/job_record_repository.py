from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.modules.job_control.domain.entity_objects.job_record import JobRecord


@runtime_checkable
class JobRecordRepository(Protocol):
    async def create_job_if_not_exists(
        self,
        *,
        job: JobRecord,
        schema_name: str | None,
        schema_version: str | None,
        message_key: str | None = None,
    ) -> bool: ...

    async def get_by_request_id(self, request_id: str) -> list[dict]: ...

    async def mark_processed_by_request_id(self, request_id: str) -> bool: ...

    async def mark_failed_by_request_id(
        self,
        *,
        request_id: str,
        error_message: str,
    ) -> bool: ...
