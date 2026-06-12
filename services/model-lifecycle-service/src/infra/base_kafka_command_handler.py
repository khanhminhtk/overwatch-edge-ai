from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from src.applications.dtos.events.kafka_base import DeadLetterEvent

if TYPE_CHECKING:
    from src.utils.logger import Logger
    from src.infra.kafka_producer import KafkaProducerClient


EventT = TypeVar("EventT", bound=BaseModel)
PayloadT = TypeVar("PayloadT", bound=BaseModel)


class BaseKafkaCommandHandler(Generic[EventT, PayloadT]):
    def __init__(
        self,
        *,
        event_type: str,
        event_model: type[EventT],
        payload_model: type[PayloadT],
        job_repository: Any,
        producer: "KafkaProducerClient",
        logger: "Logger",
        consumer_group: str,
        server_id: str,
        dlq_topic: str,
    ) -> None:
        self._event_type = event_type
        self._event_model = event_model
        self._payload_model = payload_model
        self._job_repository = job_repository
        self._producer = producer
        self._logger = logger
        self._consumer_group = consumer_group
        self._server_id = server_id
        self._dlq_topic = dlq_topic

    def handle_message(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        raw_value: bytes,
    ) -> None:
        try:
            payload = json.loads(raw_value.decode("utf-8"))
        except Exception as exc:
            self._publish_dlq(
                topic=topic,
                partition=partition,
                offset=offset,
                payload=None,
                error_message=f"Invalid JSON: {exc}",
            )
            return

        event_type = payload.get("event_type")
        if event_type != self._event_type:
            self._publish_dlq(
                topic=topic,
                partition=partition,
                offset=offset,
                payload=payload,
                error_message=f"Unsupported event_type: {event_type}",
            )
            return

        try:
            event = self._event_model.model_validate(payload)
        except ValidationError as exc:
            self._publish_dlq(
                topic=topic,
                partition=partition,
                offset=offset,
                payload=payload,
                error_message=f"Validation error: {exc}",
            )
            return

        payload_model = self._payload_model.model_validate(event.payload)

        if self.should_skip_event(event=event, payload_model=payload_model):
            return

        created = self.process_event(
            event=event,
            payload_model=payload_model,
            topic=topic,
            partition=partition,
            offset=offset,
        )

        if created:
            self.log_created(event=event, payload_model=payload_model)
            return

        self.log_duplicate(event=event)

    def should_skip_event(self, *, event: EventT, payload_model: PayloadT) -> bool:
        return False

    def process_event(
        self,
        *,
        event: EventT,
        payload_model: PayloadT,
        topic: str,
        partition: int,
        offset: int,
    ) -> bool:
        return self._job_repository.create_job_if_not_exists(
            event=event,
            topic=topic,
            partition_id=partition,
            message_offset=offset,
            consumer_group=self._consumer_group,
        )

    def log_created(self, *, event: EventT, payload_model: PayloadT) -> None:
        raise NotImplementedError

    def log_duplicate(self, *, event: EventT) -> None:
        raise NotImplementedError

    def _publish_dlq(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        payload: dict[str, Any] | None,
        error_message: str,
    ) -> None:
        dlq_event = DeadLetterEvent(
            event_type="dead_letter",
            original_topic=topic,
            original_partition=partition,
            original_offset=offset,
            payload=payload,
            error_message=error_message,
        )

        self._producer.publish(
            topic=self._dlq_topic,
            key=dlq_event.dead_letter_id,
            value=dlq_event.model_dump(mode="json"),
        )
        self._logger.error(
            "[DLQ_PUBLISHED]",
            f"original_topic={topic}",
            f"partition={partition}",
            f"offset={offset}",
            f"error={error_message}",
        )
