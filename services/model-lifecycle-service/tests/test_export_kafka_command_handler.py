import json
import unittest
from unittest.mock import Mock

from src.infra.export_kafka_command_handler import ExportKafkaCommandHandler


class ExportKafkaCommandHandlerTest(unittest.TestCase):
    def test_handle_message_publishes_dlq_for_invalid_json(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        handler = ExportKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=logger,
            consumer_group="export-consumer",
            server_id="edge-01",
            dlq_topic="export.dlq",
        )

        handler.handle_message(
            topic="export.jobs",
            partition=1,
            offset=10,
            raw_value=b"{invalid-json",
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()
        self.assertEqual(producer.publish.call_args.kwargs["topic"], "export.dlq")

    def test_handle_message_publishes_dlq_for_unsupported_event_type(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = ExportKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="export-consumer",
            server_id="edge-01",
            dlq_topic="export.dlq",
        )

        handler.handle_message(
            topic="export.jobs",
            partition=1,
            offset=11,
            raw_value=json.dumps({"event_type": "export_completed", "request_id": "req-1"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()
        payload = producer.publish.call_args.kwargs["value"]
        self.assertEqual(payload["event_type"], "dead_letter")
        self.assertEqual(payload["original_topic"], "export.jobs")

    def test_handle_message_publishes_dlq_for_validation_error(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = ExportKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="export-consumer",
            server_id="edge-01",
            dlq_topic="export.dlq",
        )

        handler.handle_message(
            topic="export.jobs",
            partition=1,
            offset=12,
            raw_value=json.dumps({"event_type": "export_requested"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()

    def test_handle_message_creates_job_for_valid_export_requested_event(self) -> None:
        repository = Mock(return_value=True)
        repository.create_job_if_not_exists.return_value = True
        producer = Mock()
        handler = ExportKafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="export-consumer",
            server_id="edge-01",
            dlq_topic="export.dlq",
        )

        handler.handle_message(
            topic="export.jobs",
            partition=2,
            offset=20,
            raw_value=json.dumps(
                {
                    "event_type": "export_requested",
                    "request_id": "req-3",
                    "payload": {
                        "version": "1.0",
                        "model_name": "recognizer",
                        "model_version": "recognizer-v2",
                    },
                }
            ).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_called_once()
        call = repository.create_job_if_not_exists.call_args
        self.assertEqual(call.kwargs["topic"], "export.jobs")
        self.assertEqual(call.kwargs["partition_id"], 2)
        self.assertEqual(call.kwargs["message_offset"], 20)
        self.assertEqual(call.kwargs["consumer_group"], "export-consumer")
        producer.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
