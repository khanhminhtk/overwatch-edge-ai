import json
import unittest
from unittest.mock import Mock, patch

from src.infra.kafka_producer import KafkaProducerClient


class KafkaProducerClientTest(unittest.TestCase):
    @patch("src.infra.kafka_producer.Producer")
    def test_init_builds_producer_with_expected_config(self, producer_cls: Mock) -> None:
        logger = Mock()
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service-server-A",
            security_protocol="PLAINTEXT",
            defaults=Mock(
                producer=Mock(
                    acks="all",
                    retries=3,
                    retry_backoff_ms=1000,
                    enable_idempotence=True,
                    linger_ms=5,
                    compression_type="snappy",
                )
            ),
        )

        client = KafkaProducerClient(logger=logger, config_loader=config_loader)

        producer_cls.assert_not_called()
        client._get_or_create_producer()
        producer_cls.assert_called_once_with(
            {
                "bootstrap.servers": "localhost:9092",
                "client.id": "model-lifecycle-service-server-A",
                "security.protocol": "PLAINTEXT",
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 1000,
                "enable.idempotence": True,
                "linger.ms": 5,
                "compression.type": "snappy",
            }
        )

    @patch("src.infra.kafka_producer.Producer")
    def test_publish_serializes_value_and_flushes(self, producer_cls: Mock) -> None:
        logger = Mock()
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service-server-A",
            security_protocol="PLAINTEXT",
            defaults=Mock(
                producer=Mock(
                    acks="all",
                    retries=3,
                    retry_backoff_ms=1000,
                    enable_idempotence=True,
                    linger_ms=5,
                    compression_type="snappy",
                )
            ),
        )
        producer = producer_cls.return_value
        client = KafkaProducerClient(logger=logger, config_loader=config_loader)

        client.publish(
            topic="model-events",
            key="model-123",
            value={"status": "ready"},
            headers={"trace_id": "req-1"},
        )

        produce_kwargs = producer.produce.call_args.kwargs
        self.assertEqual(produce_kwargs["topic"], "model-events")
        self.assertEqual(produce_kwargs["key"], b"model-123")
        self.assertEqual(produce_kwargs["value"], json.dumps({"status": "ready"}).encode("utf-8"))
        self.assertEqual(produce_kwargs["headers"], {"trace_id": "req-1"})
        self.assertTrue(callable(produce_kwargs["on_delivery"]))
        producer.flush.assert_called_once_with()

    @patch("src.infra.kafka_producer.Producer")
    def test_on_delivery_logs_error(self, producer_cls: Mock) -> None:
        logger = Mock()
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service-server-A",
            security_protocol="PLAINTEXT",
            defaults=Mock(
                producer=Mock(
                    acks="all",
                    retries=3,
                    retry_backoff_ms=1000,
                    enable_idempotence=True,
                    linger_ms=5,
                    compression_type="snappy",
                )
            ),
        )
        client = KafkaProducerClient(logger=logger, config_loader=config_loader)

        client._on_delivery(err="broker unavailable", msg=Mock())

        logger.error.assert_called_once_with(
            "[KAFKA_PRODUCER_ERROR] delivery failed: broker unavailable"
        )
        logger.info.assert_not_called()

    @patch("src.infra.kafka_producer.Producer")
    def test_on_delivery_logs_success_metadata(self, producer_cls: Mock) -> None:
        logger = Mock()
        config_loader = Mock()
        config_loader.get_typed_config.return_value = Mock(
            bootstrap_servers="localhost:9092",
            client_id_prefix="model-lifecycle-service-server-A",
            security_protocol="PLAINTEXT",
            defaults=Mock(
                producer=Mock(
                    acks="all",
                    retries=3,
                    retry_backoff_ms=1000,
                    enable_idempotence=True,
                    linger_ms=5,
                    compression_type="snappy",
                )
            ),
        )
        client = KafkaProducerClient(logger=logger, config_loader=config_loader)
        msg = Mock()
        msg.topic.return_value = "model-events"
        msg.partition.return_value = 2
        msg.offset.return_value = 17

        client._on_delivery(err=None, msg=msg)

        logger.info.assert_called_once_with(
            "[KAFKA_PRODUCER_OK]",
            "topic=model-events",
            "partition=2",
            "offset=17",
        )
        logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
