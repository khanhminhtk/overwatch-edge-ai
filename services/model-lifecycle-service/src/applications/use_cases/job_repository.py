from __future__ import annotations

from typing import Any

from psycopg import sql

from src.applications.dtos.events.kafka_base import BaseRequestedEvent
from src.infra.queries.kafka_event_queries import (
    INSERT_KAFKA_EVENT,
    SELECT_KAFKA_EVENTS_BY_REQUEST_ID,
    UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID,
    UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID,
)
from src.infra.sql import PostgresSQLHandler
from src.utils.logger import Logger


class BaseJobRepository:
    def __init__(
        self,
        *,
        sql_handler: PostgresSQLHandler,
        logger: Logger,
        schema_name: str,
    ) -> None:
        self.sql_handler = sql_handler
        self.logger = logger
        self._schema_name = schema_name

    def create_job_if_not_exists(
        self,
        event: BaseRequestedEvent,
        topic: str,
        partition_id: int,
        message_offset: int,
        consumer_group: str,
    ) -> bool:
        existing_events = self.sql_handler.execute_query(
            SELECT_KAFKA_EVENTS_BY_REQUEST_ID,
            params={"request_id": event.request_id},
        )
        if existing_events:
            self.logger.info(
                f"{self.__class__.__name__}.create_job_if_not_exists duplicate request_id={event.request_id}"
            )
            return False

        try:
            payload = self._normalize_payload(event.payload)
            self.sql_handler.execute_query(
                INSERT_KAFKA_EVENT,
                params={
                    "request_id": event.request_id,
                    "topic": topic,
                    "partition_id": partition_id,
                    "message_offset": message_offset,
                    "consumer_group": consumer_group,
                    "message_key": self._get_message_key(event),
                    "event_type": event.event_type,
                    "schema_name": self._schema_name,
                    "schema_version": self._get_schema_version(event.payload),
                    "payload": payload,
                    "status": "RECEIVED",
                    "produced_at": event.created_at,
                },
            )
            self.logger.info(
                f"{self.__class__.__name__}.create_job_if_not_exists inserted request_id={event.request_id}"
            )
            return True
        except Exception as exc:
            self.logger.error(
                f"{self.__class__.__name__}.create_job_if_not_exists failed request_id={event.request_id}: {exc}"
            )
            raise

    def claim_next_pending_job(
        self,
        *,
        server_id: str,
        query: sql.SQL | str,
        return_params: list[str],
    ) -> dict[str, Any] | None:
        try:
            result = self.sql_handler.execute_query(
                query,
                params={"server_id": server_id},
            )
            self.logger.info(
                f"{self.__class__.__name__}.claim_next_pending_job server_id={server_id} result={result}"
            )
        except Exception as exc:
            self.logger.error(
                f"{self.__class__.__name__}.claim_next_pending_job failed server_id={server_id}: {exc}"
            )
            raise

        if not result:
            return None

        row = result[0]
        if isinstance(row, dict):
            return row
        return {
            key: row[idx] for idx, key in enumerate(return_params)
        }

    def get_claim_next_pending_job(self, server_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def make_job_successful(self, request_id: str) -> None:
        try:
            self.sql_handler.execute_query(
                UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID,
                params={"request_id": request_id},
            )
            self.logger.info(f"{self.__class__.__name__}.make_job_successful request_id={request_id}")
        except Exception as exc:
            self.logger.error(
                f"{self.__class__.__name__}.make_job_successful failed request_id={request_id}: {exc}"
            )
            raise

    def make_job_failed(self, request_id: str, error_message: str) -> None:
        try:
            self.sql_handler.execute_query(
                UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID,
                params={"request_id": request_id, "error_message": error_message},
            )
            self.logger.info(f"{self.__class__.__name__}.make_job_failed request_id={request_id}")
        except Exception as exc:
            self.logger.error(
                f"{self.__class__.__name__}.make_job_failed failed request_id={request_id}: {exc}"
            )
            raise

    @staticmethod
    def _get_message_key(event: BaseRequestedEvent) -> str | None:
        payload = event.payload
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload.get("model_name")
        return getattr(payload, "model_name", None)

    @staticmethod
    def _normalize_payload(payload: Any | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "get_dict"):
            return payload.get_dict()
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    @staticmethod
    def _get_schema_version(payload: Any | None) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            version = payload.get("version")
            return str(version) if version is not None else None
        version = getattr(payload, "version", None)
        return str(version) if version is not None else None
