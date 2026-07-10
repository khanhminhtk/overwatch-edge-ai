from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto


@runtime_checkable
class JobHandler(Protocol):
    async def handle(self, job: ClaimedJobDto) -> JobResultDto: ...
