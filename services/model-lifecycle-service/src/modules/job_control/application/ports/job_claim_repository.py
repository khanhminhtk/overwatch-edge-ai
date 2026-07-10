from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto


@runtime_checkable
class JobClaimRepository(Protocol):
    async def claim_next_pending_job(
        self,
        *,
        server_id: str,
    ) -> ClaimedJobDto | None: ...
