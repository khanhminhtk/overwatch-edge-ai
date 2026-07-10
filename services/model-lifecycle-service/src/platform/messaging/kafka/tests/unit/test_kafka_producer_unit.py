from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from src.platform.messaging.kafka.producer import KafkaProducerClient


class KafkaProducerClientInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.bootstrap_servers = "broker:9092"
        self.config.security_protocol = "PLAINTEXT"
        self.config.client_id_prefix = "prod-svc"
        self.config.producer_config.acks = "all"
        self.config.producer_config.enable_idempotence = True
        self.config.producer_config.retries = 3
        self.config.producer_config.linger_ms = 5
        self.config.producer_config.retry_backoff_ms = 1000
        self.config.producer_config.compression_type = "snappy"

        self.logger = MagicMock()

        patcher = patch(
            "src.platform.messaging.kafka.producer.ConfluentProducer"
        )
        self.mock_producer_class = patcher.start()
        self.mock_producer = MagicMock()
        self.mock_producer_class.return_value = self.mock_producer
        self.addCleanup(patcher.stop)

        self.client = KafkaProducerClient(
            config=self.config,
            logger=self.logger,
        )

    def test_init_creates_producer_with_config(self) -> None:
        self.mock_producer_class.assert_called_once_with(
            KafkaProducerClient._to_confluent_config(self.config)
        )

    def test_producer_stored_as_attribute(self) -> None:
        self.assertIs(self.client._producer, self.mock_producer)


class KafkaProducerClientPublishTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.bootstrap_servers = "broker:9092"
        self.config.security_protocol = "PLAINTEXT"
        self.config.client_id_prefix = "prod-svc"
        self.config.producer_config.acks = "all"
        self.config.producer_config.enable_idempotence = True
        self.config.producer_config.retries = 3
        self.config.producer_config.linger_ms = 5
        self.config.producer_config.retry_backoff_ms = 1000
        self.config.producer_config.compression_type = "snappy"

        self.logger = MagicMock()

        patcher = patch(
            "src.platform.messaging.kafka.producer.ConfluentProducer"
        )
        self.mock_producer_class = patcher.start()
        self.mock_producer = MagicMock()
        self.mock_producer_class.return_value = self.mock_producer
        self.addCleanup(patcher.stop)

        self.client = KafkaProducerClient(
            config=self.config,
            logger=self.logger,
        )

    def test_publish_calls_produce_and_flush(self) -> None:
        self.client.publish(
            topic="my-topic",
            key=b"my-key",
            value=b"my-value",
        )
        self.mock_producer.produce.assert_called_once()
        self.mock_producer.flush.assert_called_once()

    def test_publish_passes_correct_arguments(self) -> None:
        self.client.publish(
            topic="my-topic",
            key=b"k",
            value=b"v",
            headers=[("h1", b"v1")],
        )
        call_kwargs = self.mock_producer.produce.call_args.kwargs
        self.assertEqual(call_kwargs["topic"], "my-topic")
        self.assertEqual(call_kwargs["key"], b"k")
        self.assertEqual(call_kwargs["value"], b"v")
        self.assertEqual(call_kwargs["headers"], [("h1", b"v1")])
        self.assertEqual(
            call_kwargs["on_delivery"], self.client._delivery_report
        )

    def test_publish_with_none_key_and_value(self) -> None:
        self.client.publish(topic="t", key=None, value=None)
        call_kwargs = self.mock_producer.produce.call_args.kwargs
        self.assertIsNone(call_kwargs["key"])
        self.assertIsNone(call_kwargs["value"])

    def test_publish_without_headers(self) -> None:
        self.client.publish(topic="t", key=None, value=b"v")
        call_kwargs = self.mock_producer.produce.call_args.kwargs
        self.assertIsNone(call_kwargs["headers"])

    def test_publish_re_raises_on_produce_failure(self) -> None:
        self.mock_producer.produce.side_effect = RuntimeError("kafka down")
        with self.assertRaises(RuntimeError):
            self.client.publish(
                topic="t", key=None, value=b"v"
            )

    def test_publish_re_raises_on_flush_failure(self) -> None:
        self.mock_producer.flush.side_effect = RuntimeError("flush failed")
        with self.assertRaises(RuntimeError):
            self.client.publish(
                topic="t", key=None, value=b"v"
            )

    def test_publish_logs_error_on_failure(self) -> None:
        self.mock_producer.produce.side_effect = RuntimeError("err")
        with self.assertRaises(RuntimeError):
            self.client.publish(
                topic="fail-topic", key=None, value=b"v"
            )
        self.logger.error.assert_called_once()
        log_arg = self.logger.error.call_args[0][0]
        self.assertIn("PRODUCER_ERROR", log_arg)
        self.assertIn("fail-topic", log_arg)


class KafkaProducerClientDeliveryReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.producer_config.acks = "all"
        self.config.producer_config.enable_idempotence = True
        self.config.producer_config.retries = 3
        self.config.producer_config.linger_ms = 5
        self.config.producer_config.retry_backoff_ms = 1000
        self.config.producer_config.compression_type = "snappy"

        self.logger = MagicMock()

        patcher = patch(
            "src.platform.messaging.kafka.producer.ConfluentProducer"
        )
        self.mock_producer_class = patcher.start()
        self.mock_producer = MagicMock()
        self.mock_producer_class.return_value = self.mock_producer
        self.addCleanup(patcher.stop)

        self.client = KafkaProducerClient(
            config=self.config,
            logger=self.logger,
        )

    def test_delivery_report_success_logs_info(self) -> None:
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "t"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 99
        self.client._delivery_report(None, mock_msg)
        self.logger.info.assert_called_once()
        log_arg = self.logger.info.call_args[0][0]
        self.assertIn("DELIVERY_REPORT_SUCCESS", log_arg)

    def test_delivery_report_error_logs_error(self) -> None:
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "t"
        err = Exception("broker offline")
        self.client._delivery_report(err, mock_msg)
        self.logger.error.assert_called_once()
        log_arg = self.logger.error.call_args[0][0]
        self.assertIn("DELIVERY_REPORT_ERROR", log_arg)


class KafkaProducerClientToConfluentConfigTest(unittest.TestCase):
    def test_returns_correct_dict(self) -> None:
        config = MagicMock()
        config.bootstrap_servers = "host:9092"
        config.security_protocol = "SASL_SSL"
        config.client_id_prefix = "my-prod"
        config.producer_config.acks = "1"
        config.producer_config.enable_idempotence = False
        config.producer_config.retries = 10
        config.producer_config.linger_ms = 0
        config.producer_config.retry_backoff_ms = 500
        config.producer_config.compression_type = "lz4"

        result = KafkaProducerClient._to_confluent_config(config)
        expected = {
            "bootstrap.servers": "host:9092",
            "security.protocol": "SASL_SSL",
            "client.id": "my-prod",
            "acks": "1",
            "enable.idempotence": False,
            "retries": 10,
            "linger.ms": 0,
            "retry.backoff.ms": 500,
            "compression.type": "lz4",
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
