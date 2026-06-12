from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.infra.kafka_consumer import KafkaConsumerClient


class KafkaConsumerClientTest(unittest.TestCase):
    @patch("src.infra.kafka_consumer.Consumer")
    def test_init_builds_consumer_for_multiple_job_keys(self, consumer_cls: Mock) -> None:
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service",
            security_protocol="PLAINTEXT",
            jobs={
                "train_recognizer": Mock(
                    topic="training.recognizer",
                    group_id="training-group",
                    consumer=Mock(
                        auto_offset_reset="earliest",
                        enable_auto_commit=False,
                        session_timeout_ms=45000,
                        max_poll_interval_ms=300000,
                    ),
                ),
                "train_detection": Mock(
                    topic="training.detection",
                    group_id="training-group",
                    consumer=Mock(
                        auto_offset_reset="earliest",
                        enable_auto_commit=False,
                        session_timeout_ms=45000,
                        max_poll_interval_ms=300000,
                    ),
                ),
            },
        )

        client = KafkaConsumerClient(
            config_loader=config_loader,
            job_keys=["train_recognizer", "train_detection"],
            logger=Mock(),
        )

        self.assertEqual(client._topics, ["training.recognizer", "training.detection"])
        consumer_cls.assert_called_once()
        self.assertEqual(
            consumer_cls.call_args.args[0]["group.id"],
            "training-group",
        )

    @patch("src.infra.kafka_consumer.Consumer")
    def test_init_rejects_mixed_consumer_groups(self, consumer_cls: Mock) -> None:
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service",
            security_protocol="PLAINTEXT",
            jobs={
                "train_recognizer": Mock(
                    topic="training.recognizer",
                    group_id="group-a",
                    consumer=Mock(
                        auto_offset_reset="earliest",
                        enable_auto_commit=False,
                        session_timeout_ms=45000,
                        max_poll_interval_ms=300000,
                    ),
                ),
                "export_recognizer_tensorrt": Mock(
                    topic="export.recognizer",
                    group_id="group-b",
                    consumer=Mock(
                        auto_offset_reset="earliest",
                        enable_auto_commit=False,
                        session_timeout_ms=45000,
                        max_poll_interval_ms=300000,
                    ),
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "All job_keys must share the same consumer group"):
            KafkaConsumerClient(
                config_loader=config_loader,
                job_keys=["train_recognizer", "export_recognizer_tensorrt"],
                logger=Mock(),
            )

        consumer_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
