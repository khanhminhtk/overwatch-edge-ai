from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.infra.topic_routing_kafka_handler import TopicRoutingKafkaHandler


class TopicRoutingKafkaHandlerTest(unittest.TestCase):
    def test_handle_message_routes_by_topic(self) -> None:
        training_handler = Mock()
        export_handler = Mock()
        router = TopicRoutingKafkaHandler(
            handlers_by_topic={
                "training.jobs": training_handler,
                "export.jobs": export_handler,
            }
        )

        router.handle_message(
            topic="export.jobs",
            partition=1,
            offset=9,
            raw_value=b"{}",
        )

        training_handler.handle_message.assert_not_called()
        export_handler.handle_message.assert_called_once_with(
            topic="export.jobs",
            partition=1,
            offset=9,
            raw_value=b"{}",
        )

    def test_init_rejects_duplicate_topics(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate topic handler registration"):
            TopicRoutingKafkaHandler.from_topic_handlers(
                [
                    ("training.jobs", Mock()),
                    ("training.jobs", Mock()),
                ]
            )

    def test_handle_message_rejects_unregistered_topic(self) -> None:
        router = TopicRoutingKafkaHandler(handlers_by_topic={"training.jobs": Mock()})

        with self.assertRaisesRegex(ValueError, "No handler registered for topic"):
            router.handle_message(
                topic="export.jobs",
                partition=1,
                offset=9,
                raw_value=b"{}",
            )


if __name__ == "__main__":
    unittest.main()
