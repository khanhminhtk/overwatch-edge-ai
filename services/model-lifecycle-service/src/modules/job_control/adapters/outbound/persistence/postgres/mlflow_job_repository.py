from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psycopg import sql

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_claim_repository import (
    PostgresJobClaimRepository,
)
from src.platform.logger import Logger

if TYPE_CHECKING:
    from src.platform.persistence.postgres.transaction import PostgresTransaction


DEFAULT_MLFLOW_RETURNING = """
e.id,
e.request_id,
e.payload->>'model_name' AS model_name,
e.payload->>'model_version' AS model_version,
e.payload->>'git_commit' AS git_commit,
e.payload->>'checkpoint_best_name' AS checkpoint_best_name,
e.payload->>'checkpoint_last_name' AS checkpoint_last_name,
e.payload->>'event' AS event,
e.payload->>'version' AS version,
e.status
"""


class MLflowJobRepository:
    def __init__(
        self,
        *,
        transaction: "PostgresTransaction",
        logger: Logger,
        event_type: str,
        reclaim_timeout_seconds: int = 1800,
        event_filter: str | None = None,
        model_name_filter: str | None = None,
    ) -> None:
        self._claim_repository = PostgresJobClaimRepository(
            transaction=transaction,
            logger=logger,
        )
        self._event_type = event_type
        self._reclaim_timeout_seconds = reclaim_timeout_seconds
        self._event_filter = event_filter
        self._model_name_filter = model_name_filter

    async def claim_next_pending_job(self, *, server_id: str) -> ClaimedJobDto | None:
        row = await self._claim_repository.claim_next_pending_job(
            server_id=server_id,
            event_type=self._event_type,
            reclaim_timeout_seconds=self._reclaim_timeout_seconds,
            additional_conditions=self._build_additional_conditions(),
            returning=sql.SQL(DEFAULT_MLFLOW_RETURNING),
        )
        if row is None:
            return None
        return ClaimedJobDto(
            request_id=str(row["request_id"]),
            event_type=self._event_type,
            payload={
                "model_name": row.get("model_name"),
                "model_version": row.get("model_version"),
                "git_commit": row.get("git_commit"),
                "checkpoint_best_name": row.get("checkpoint_best_name"),
                "checkpoint_last_name": row.get("checkpoint_last_name"),
                "event": row.get("event"),
                "version": row.get("version"),
            },
            status=str(row["status"]),
        )

    def _build_additional_conditions(self) -> sql.SQL:
        additional_conditions = sql.SQL("")
        if self._event_filter:
            additional_conditions = additional_conditions + sql.SQL(
                " AND e.payload->>'event' = {event_filter}"
            ).format(event_filter=sql.Literal(self._event_filter))
        if self._model_name_filter:
            additional_conditions = additional_conditions + sql.SQL(
                " AND e.payload->>'model_name' = {model_name_filter}"
            ).format(model_name_filter=sql.Literal(self._model_name_filter))
        return additional_conditions
