from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from pydantic import BaseModel

from src.infra.base_kafka_command_handler import BaseKafkaCommandHandler


class _PayloadModel(BaseModel):
    model_name: str


class _EventModel(BaseModel):
    event_type: str
    request_id: str
    payload: _PayloadModel


class _TestKafkaCommandHandler(BaseKafkaCommandHandler[_EventModel, _PayloadModel]):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            event_type="test_requested",
            event_model=_EventModel,
            payload_model=_PayloadModel,
            job_repository=kwargs["job_repository"],
            producer=kwargs["producer"],
            logger=kwargs["logger"],
            consumer_group=kwargs["consumer_group"],
            server_id=kwargs["server_id"],
            dlq_topic=kwargs["dlq_topic"],
        )
        self.process_event = Mock(return_value=True)
        self.log_created = Mock()
        self.log_duplicate = Mock()


class BaseKafkaCommandHandlerTest(unittest.TestCase):
    def test_handle_message_publishes_dlq_for_invalid_json(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = _TestKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="test-consumer",
            server_id="edge-01",
            dlq_topic="test.dlq",
        )

        handler.handle_message(
            topic="test.jobs",
            partition=1,
            offset=10,
            raw_value=b"{invalid-json",
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()
        self.assertEqual(producer.publish.call_args.kwargs["topic"], "test.dlq")

    def test_handle_message_publishes_dlq_for_unsupported_event_type(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = _TestKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="test-consumer",
            server_id="edge-01",
            dlq_topic="test.dlq",
        )

        handler.handle_message(
            topic="test.jobs",
            partition=1,
            offset=11,
            raw_value=json.dumps({"event_type": "other", "request_id": "req-1"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()

    def test_handle_message_publishes_dlq_for_validation_error(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = _TestKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="test-consumer",
            server_id="edge-01",
            dlq_topic="test.dlq",
        )

        handler.handle_message(
            topic="test.jobs",
            partition=1,
            offset=12,
            raw_value=json.dumps({"event_type": "test_requested"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()

    def test_handle_message_creates_job_for_valid_event(self) -> None:
        repository = Mock()
        repository.create_job_if_not_exists.return_value = True
        producer = Mock()
        handler = _TestKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="test-consumer",
            server_id="edge-01",
            dlq_topic="test.dlq",
        )

        handler.handle_message(
            topic="test.jobs",
            partition=2,
            offset=20,
            raw_value=json.dumps(
                {
                    "event_type": "test_requested",
                    "request_id": "req-3",
                    "payload": {"model_name": "recognizer"},
                }
            ).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        handler.process_event.assert_called_once()
        handler.log_created.assert_called_once()
        producer.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
