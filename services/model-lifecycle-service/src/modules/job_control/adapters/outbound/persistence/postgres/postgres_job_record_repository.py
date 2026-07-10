from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.modules.job_control.adapters.outbound.persistence.postgres.asyncpg_query import (
    bind_query,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries import (
    INSERT_JOB_EVENT,
    MARK_JOB_EVENT_FAILED,
    MARK_JOB_EVENT_PROCESSING,
    MARK_JOB_EVENT_PROCESSED,
    SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY,
    SELECT_JOB_EVENTS_BY_REQUEST_ID,
    SELECT_JOB_EVENTS_BY_STATUS,
    UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID,
    UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID,
)
from src.modules.job_control.domain.entity_objects.job_record import JobRecord
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity
from src.platform.logger import Logger

if TYPE_CHECKING:
    from src.platform.persistence.postgres.transaction import PostgresTransaction


class PostgresJobRecordRepository:
    def __init__(
        self,
        *,
        transaction: "PostgresTransaction",
        logger: Logger,
    ) -> None:
        self._transaction = transaction
        self._logger = logger

    async def create_job_if_not_exists(
        self,
        *,
        job: JobRecord,
        schema_name: str | None,
        schema_version: str | None,
        message_key: str | None = None,
    ) -> bool:
        params = self._build_insert_params(
            job=job,
            schema_name=schema_name,
            schema_version=schema_version,
            message_key=message_key,
        )
        async with self._transaction.transaction() as connection:
            query, args = bind_query(INSERT_JOB_EVENT, params)
            row = await connection.fetchrow(query, *args)

        created = row is not None
        if created:
            self._logger.info("[JOB_EVENT_CREATED]", f"request_id={job.request_id}")
        else:
            self._logger.info("[JOB_EVENT_DUPLICATE_SKIP]", f"request_id={job.request_id}")
        return created

    async def get_by_request_id(self, request_id: str) -> list[dict[str, Any]]:
        async with self._transaction.transaction() as connection:
            query, args = bind_query(
                SELECT_JOB_EVENTS_BY_REQUEST_ID,
                {"request_id": request_id},
            )
            rows = await connection.fetch(query, *args)
        return [dict(row) for row in rows]

    async def get_by_message_identity(
        self,
        message_identity: MessageIdentity,
    ) -> dict[str, Any] | None:
        async with self._transaction.transaction() as connection:
            params = self._message_identity_params(message_identity)
            query, args = bind_query(SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY, params)
            row = await connection.fetchrow(query, *args)
        return dict(row) if row is not None else None

    async def list_by_status(
        self,
        *,
        consumer_group: str,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._transaction.transaction() as connection:
            query, args = bind_query(
                SELECT_JOB_EVENTS_BY_STATUS,
                {
                    "consumer_group": consumer_group,
                    "status": status,
                    "limit": limit,
                },
            )
            rows = await connection.fetch(query, *args)
        return [dict(row) for row in rows]

    async def mark_processing(self, message_identity: MessageIdentity) -> bool:
        return await self._update_by_message_identity(
            query=MARK_JOB_EVENT_PROCESSING,
            message_identity=message_identity,
        )

    async def mark_processed(self, message_identity: MessageIdentity) -> bool:
        return await self._update_by_message_identity(
            query=MARK_JOB_EVENT_PROCESSED,
            message_identity=message_identity,
        )

    async def mark_failed(
        self,
        *,
        message_identity: MessageIdentity,
        error_message: str,
    ) -> bool:
        params = self._message_identity_params(message_identity)
        params["error_message"] = error_message
        async with self._transaction.transaction() as connection:
            query, args = bind_query(MARK_JOB_EVENT_FAILED, params)
            row = await connection.fetchrow(query, *args)
        return row is not None

    async def mark_processed_by_request_id(self, request_id: str) -> bool:
        async with self._transaction.transaction() as connection:
            query, args = bind_query(
                UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID,
                {"request_id": request_id},
            )
            row = await connection.fetchrow(query, *args)
        return row is not None

    async def mark_failed_by_request_id(
        self,
        *,
        request_id: str,
        error_message: str,
    ) -> bool:
        async with self._transaction.transaction() as connection:
            query, args = bind_query(
                UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID,
                {
                    "request_id": request_id,
                    "error_message": error_message,
                },
            )
            row = await connection.fetchrow(query, *args)
        return row is not None

    async def _update_by_message_identity(
        self,
        *,
        query: str,
        message_identity: MessageIdentity,
    ) -> bool:
        async with self._transaction.transaction() as connection:
            params = self._message_identity_params(message_identity)
            bound_query, args = bind_query(query, params)
            row = await connection.fetchrow(bound_query, *args)
        return row is not None

    @staticmethod
    def _message_identity_params(message_identity: MessageIdentity) -> dict[str, Any]:
        return {
            "topic": message_identity.topic,
            "partition_id": message_identity.partition_id,
            "message_offset": message_identity.message_offset,
            "consumer_group": message_identity.consumer_group,
        }

    def _build_insert_params(
        self,
        *,
        job: JobRecord,
        schema_name: str | None,
        schema_version: str | None,
        message_key: str | None,
    ) -> dict[str, Any]:
        return {
            "request_id": job.request_id,
            "topic": job.message_identity.topic,
            "partition_id": job.message_identity.partition_id,
            "message_offset": job.message_identity.message_offset,
            "consumer_group": job.message_identity.consumer_group,
            "message_key": message_key,
            "event_type": job.event_type,
            "schema_name": schema_name,
            "schema_version": schema_version,
            "payload": json.dumps(job.payload) if job.payload is not None else None,
            "status": job.status.value,
            "produced_at": job.produced_at,
        }
