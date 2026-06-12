from __future__ import annotations

from typing import Any, Iterable


class TopicRoutingKafkaHandler:
    def __init__(self, *, handlers_by_topic: dict[str, Any]) -> None:
        self._handlers_by_topic = dict(handlers_by_topic)

    @classmethod
    def from_topic_handlers(cls, topic_handlers: Iterable[tuple[str, Any]]) -> "TopicRoutingKafkaHandler":
        handlers_by_topic: dict[str, Any] = {}
        for topic, handler in topic_handlers:
            if topic in handlers_by_topic:
                raise ValueError(f"Duplicate topic handler registration: {topic}")
            handlers_by_topic[topic] = handler
        return cls(handlers_by_topic=handlers_by_topic)

    def handle_message(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        raw_value: bytes,
    ) -> None:
        handler = self._handlers_by_topic.get(topic)
        if handler is None:
            raise ValueError(f"No handler registered for topic: {topic}")
        handler.handle_message(
            topic=topic,
            partition=partition,
            offset=offset,
            raw_value=raw_value,
        )
