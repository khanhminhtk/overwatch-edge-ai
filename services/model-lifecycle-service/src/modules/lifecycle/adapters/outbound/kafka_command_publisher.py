from __future__ import annotations

import json

from src.modules.lifecycle.application import LifecycleCommand
from src.platform.messaging.kafka.protocols import MessageProducer


class LifecycleKafkaCommandPublisher:
    """Publishes a stage command to the configured worker topic."""

    def __init__(self, producer: MessageProducer, topics: dict[str, str]) -> None:
        self._producer = producer
        self._topics = topics

    def publish(self, command: LifecycleCommand) -> None:
        topic = self._topics.get(command.event_type)
        if topic is None:
            raise ValueError(f"No Kafka topic configured for lifecycle event {command.event_type}")
        self._producer.publish(
            topic=topic,
            key=command.request_id.encode(),
            value=json.dumps({
                "request_id": command.request_id,
                "event_type": command.event_type,
                "payload": command.payload,
            }).encode(),
            headers=[("content-type", b"application/json")],
        )
