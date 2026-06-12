import json
import unittest
from unittest.mock import Mock

from src.infra.kafka_command_handler import KafkaCommandHandler


class KafkaCommandHandlerTest(unittest.TestCase):
    def test_handle_message_publishes_dlq_for_invalid_json(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        handler = KafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=logger,
            consumer_group="training-consumer",
            server_id="edge-01",
            dlq_topic="training.dlq",
        )

        handler.handle_message(
            topic="training.jobs",
            partition=1,
            offset=10,
            raw_value=b"{invalid-json",
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()
        self.assertEqual(producer.publish.call_args.kwargs["topic"], "training.dlq")

    def test_handle_message_publishes_dlq_for_unsupported_event_type(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = KafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="training-consumer",
            server_id="edge-01",
            dlq_topic="training.dlq",
        )

        handler.handle_message(
            topic="training.jobs",
            partition=1,
            offset=11,
            raw_value=json.dumps({"event_type": "training_completed", "request_id": "req-1"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()
        payload = producer.publish.call_args.kwargs["value"]
        self.assertEqual(payload["event_type"], "dead_letter")
        self.assertEqual(payload["original_topic"], "training.jobs")

    def test_handle_message_publishes_dlq_for_validation_error(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = KafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="training-consumer",
            server_id="edge-01",
            dlq_topic="training.dlq",
        )

        handler.handle_message(
            topic="training.jobs",
            partition=1,
            offset=12,
            raw_value=json.dumps({"event_type": "train_requested"}).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_called_once()

    def test_handle_message_skips_when_target_server_does_not_match(self) -> None:
        repository = Mock()
        producer = Mock()
        handler = KafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="training-consumer",
            server_id="edge-01",
            dlq_topic="training.dlq",
        )

        handler.handle_message(
            topic="training.jobs",
            partition=1,
            offset=13,
            raw_value=json.dumps(
                {
                    "event_type": "train_requested",
                    "request_id": "req-2",
                    "payload": {
                        "version": "1.0",
                        "model_name": "detector-a",
                        "dataset_version": "dataset-v1",
                        "target_server_id": "edge-99",
                    },
                }
            ).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_not_called()
        producer.publish.assert_not_called()

    def test_handle_message_creates_job_for_valid_train_requested_event(self) -> None:
        repository = Mock(return_value=True)
        repository.create_job_if_not_exists.return_value = True
        producer = Mock()
        handler = KafkaCommandHandler(
            job_repository=repository,
            producer=producer,
            logger=Mock(),
            consumer_group="training-consumer",
            server_id="edge-01",
            dlq_topic="training.dlq",
        )

        handler.handle_message(
            topic="training.jobs",
            partition=2,
            offset=20,
            raw_value=json.dumps(
                {
                    "event_type": "train_requested",
                    "request_id": "req-3",
                    "payload": {
                        "version": "1.0",
                        "model_name": "detector-b",
                        "dataset_version": "dataset-v2",
                        "train_config_uri": "s3://cfgs/train.yaml",
                        "target_server_id": "edge-01",
                    },
                }
            ).encode("utf-8"),
        )

        repository.create_job_if_not_exists.assert_called_once()
        call = repository.create_job_if_not_exists.call_args
        self.assertEqual(call.kwargs["topic"], "training.jobs")
        self.assertEqual(call.kwargs["partition_id"], 2)
        self.assertEqual(call.kwargs["message_offset"], 20)
        self.assertEqual(call.kwargs["consumer_group"], "training-consumer")
        producer.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
