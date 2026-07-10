from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from psycopg import sql

from src.modules.job_control.adapters.outbound.persistence.postgres.asyncpg_query import (
    bind_query,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries import (
    CLAIM_NEXT_PENDING_JOB,
)
from src.modules.job_control.domain.value_objects.local_time import local_now
from src.platform.logger import Logger

if TYPE_CHECKING:
    from src.platform.persistence.postgres.transaction import PostgresTransaction


class PostgresJobClaimRepository:
    def __init__(
        self,
        *,
        transaction: "PostgresTransaction",
        logger: Logger,
    ) -> None:
        self._transaction = transaction
        self._logger = logger

    async def claim_next_pending_job(
        self,
        *,
        server_id: str,
        event_type: str,
        reclaim_timeout_seconds: int,
        returning: sql.SQL,
        additional_conditions: sql.SQL | None = None,
    ) -> dict[str, Any] | None:
        query = self._render_claim_query(
            event_type=event_type,
            returning=returning,
            additional_conditions=additional_conditions or sql.SQL(""),
        )
        stale_cutoff = local_now() - timedelta(seconds=reclaim_timeout_seconds)
        async with self._transaction.transaction() as connection:
            bound_query, args = bind_query(
                query,
                {
                    "server_id": server_id,
                    "stale_cutoff": stale_cutoff,
                },
            )
            row = await connection.fetchrow(bound_query, *args)

        claimed = dict(row) if row is not None else None
        log_args: list[str] = [
            f"event_type={event_type}",
            f"server_id={server_id}",
            f"stale_cutoff={stale_cutoff.isoformat()}",
            f"claimed={claimed is not None}",
        ]
        if claimed is not None:
            if "request_id" in claimed:
                log_args.append(f"request_id={claimed['request_id']}")
            if "status" in claimed:
                log_args.append(f"status={claimed['status']}")
        self._logger.info("[JOB_EVENT_CLAIM_ATTEMPT]", *log_args)
        return claimed

    def _render_claim_query(
        self,
        *,
        event_type: str,
        returning: sql.SQL,
        additional_conditions: sql.SQL,
    ) -> str:
        query = CLAIM_NEXT_PENDING_JOB.format(
            event_type=sql.Literal(event_type),
            returning=returning,
            additional_conditions=additional_conditions,
        )
        return self._render_sql(query)

    def _render_sql(self, clause: sql.Composable) -> str:
        if isinstance(clause, sql.SQL):
            return clause._obj
        if isinstance(clause, sql.Literal):
            value = clause._obj
            if value is None:
                return "NULL"
            if isinstance(value, str):
                return "'" + value.replace("'", "''") + "'"
            return str(value)
        if isinstance(clause, sql.Composed):
            return "".join(self._render_sql(part) for part in clause)
        raise TypeError(f"Unsupported SQL clause type: {type(clause)!r}")
