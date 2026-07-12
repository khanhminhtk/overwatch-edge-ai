from __future__ import annotations

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.ports.job_claim_repository import (
    JobClaimRepository,
)


class ClaimNextPendingJob:
    def __init__(self, *, repository: JobClaimRepository) -> None:
        self._repository = repository

    async def execute(self, *, server_id: str) -> ClaimedJobDto | None:
        return await self._repository.claim_next_pending_job(server_id=server_id)
