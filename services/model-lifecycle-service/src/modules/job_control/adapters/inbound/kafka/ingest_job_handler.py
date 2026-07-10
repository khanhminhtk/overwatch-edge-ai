from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.modules.job_control.application.dto.consume_event_command import (
    ConsumeEventCommand,
)
from src.modules.job_control.application.use_case.ingest_job import (
    IngestJob,
)
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity
from src.platform.messaging.kafka.message import ConsumedMessage

if TYPE_CHECKING:
    from src.platform.logger import Logger


class IngestJobHandler:
    def __init__(
        self,
        *,
        ingest_job: IngestJob,
        logger: "Logger",
        consumer_group: str,
        schema_name: str = "mlflow_tracking_job",
        schema_version: str = "1.0",
    ) -> None:
        self._ingest_job = ingest_job
        self._logger = logger
        self._consumer_group = consumer_group
        self._schema_name = schema_name
        self._schema_version = schema_version

    async def handle(self, message: ConsumedMessage) -> bool:
        payload = self._decode_message(message)
        event_type = self._require_string(payload, "event_type")
        request_id = self._require_string(payload, "request_id")
        body = payload.get("payload")
        if body is not None and not isinstance(body, dict):
            raise ValueError("Kafka payload.payload must be an object when present")

        command = ConsumeEventCommand(
            request_id=request_id,
            message_identity=MessageIdentity(
                topic=message.topic,
                partition_id=message.partition,
                message_offset=message.offset,
                consumer_group=self._consumer_group,
            ),
            event_type=event_type,
            payload=body,
            schema_name=self._schema_name,
            schema_version=self._schema_version,
            message_key=self._decode_key(message.key),
        )
        created = await self._ingest_job.execute(command)
        self._logger.info(
            f"[{type(self).__name__}]",
            f"request_id={request_id}",
            f"event_type={event_type}",
            f"created={created}",
        )
        return created

    @staticmethod
    def _decode_message(message: ConsumedMessage) -> dict[str, Any]:
        try:
            data = json.loads(message.require_value().decode("utf-8"))
        except Exception as exc:
            raise ValueError("Kafka message is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Kafka message body must be a JSON object")
        return data

    @staticmethod
    def _require_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Kafka payload field {key} must be a non-empty string")
        return value

    @staticmethod
    def _decode_key(key: bytes | None) -> str | None:
        if key is None:
            return None
        return key.decode("utf-8")
