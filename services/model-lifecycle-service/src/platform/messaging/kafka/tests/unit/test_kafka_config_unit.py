from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from src.platform.messaging.kafka.config import (
    KafkaConfig,
    KafkaConsumerConfig,
    KafkaJobConfig,
    KafkaProducerConfig,
)


class KafkaConsumerConfigTest(unittest.TestCase):
    def test_creates_with_valid_fields(self) -> None:
        config = KafkaConsumerConfig(
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            session_timeout_ms=45000,
            max_poll_interval_ms=300000,
        )

        self.assertEqual(config.auto_offset_reset, "earliest")
        self.assertFalse(config.enable_auto_commit)
        self.assertEqual(config.session_timeout_ms, 45000)
        self.assertEqual(config.max_poll_interval_ms, 300000)

    def test_is_frozen(self) -> None:
        config = KafkaConsumerConfig(
            auto_offset_reset="latest",
            enable_auto_commit=True,
            session_timeout_ms=30000,
            max_poll_interval_ms=600000,
        )

        with self.assertRaises(FrozenInstanceError):
            config.auto_offset_reset = "none"  # type: ignore[misc]

    def test_two_instances_are_equal_when_fields_match(self) -> None:
        kwargs = dict(
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            session_timeout_ms=45000,
            max_poll_interval_ms=300000,
        )

        self.assertEqual(KafkaConsumerConfig(**kwargs), KafkaConsumerConfig(**kwargs))


class KafkaProducerConfigTest(unittest.TestCase):
    def test_creates_with_valid_fields(self) -> None:
        config = KafkaProducerConfig(
            acks="all",
            enable_idempotence=True,
            retries=3,
            linger_ms=5,
            retry_backoff_ms=1000,
            compression_type="snappy",
        )

        self.assertEqual(config.acks, "all")
        self.assertTrue(config.enable_idempotence)
        self.assertEqual(config.retries, 3)
        self.assertEqual(config.linger_ms, 5)
        self.assertEqual(config.retry_backoff_ms, 1000)
        self.assertEqual(config.compression_type, "snappy")

    def test_is_frozen(self) -> None:
        config = KafkaProducerConfig(
            acks="all",
            enable_idempotence=True,
            retries=3,
            linger_ms=5,
            retry_backoff_ms=1000,
            compression_type="none",
        )

        with self.assertRaises(FrozenInstanceError):
            config.retries = 10  # type: ignore[misc]

    def test_accepts_all_compression_types(self) -> None:
        for ct in ("none", "gzip", "snappy", "lz4", "zstd"):
            config = KafkaProducerConfig(
                acks="all",
                enable_idempotence=True,
                retries=1,
                linger_ms=0,
                retry_backoff_ms=100,
                compression_type=ct,
            )
            self.assertEqual(config.compression_type, ct)


class KafkaConfigTest(unittest.TestCase):
    def _make_consumer(self) -> KafkaConsumerConfig:
        return KafkaConsumerConfig(
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            session_timeout_ms=45000,
            max_poll_interval_ms=300000,
        )

    def _make_producer(self) -> KafkaProducerConfig:
        return KafkaProducerConfig(
            acks="all",
            enable_idempotence=True,
            retries=3,
            linger_ms=5,
            retry_backoff_ms=1000,
            compression_type="snappy",
        )

    def test_creates_with_valid_fields(self) -> None:
        consumer = self._make_consumer()
        producer = self._make_producer()

        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            security_protocol="PLAINTEXT",
            client_id_prefix="test-service",
            consumer_config=consumer,
            producer_config=producer,
        )

        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.security_protocol, "PLAINTEXT")
        self.assertEqual(config.client_id_prefix, "test-service")
        self.assertIs(config.consumer_config, consumer)
        self.assertIs(config.producer_config, producer)

    def test_is_frozen(self) -> None:
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            security_protocol="PLAINTEXT",
            client_id_prefix="test-service",
            consumer_config=self._make_consumer(),
            producer_config=self._make_producer(),
        )

        with self.assertRaises(FrozenInstanceError):
            config.bootstrap_servers = "other:9093"  # type: ignore[misc]

    def test_nested_configs_are_independent(self) -> None:
        c1 = self._make_consumer()
        c2 = KafkaConsumerConfig(
            auto_offset_reset="latest",
            enable_auto_commit=True,
            session_timeout_ms=30000,
            max_poll_interval_ms=600000,
        )

        cfg1 = KafkaConfig(
            bootstrap_servers="host1:9092",
            security_protocol="PLAINTEXT",
            client_id_prefix="svc1",
            consumer_config=c1,
            producer_config=self._make_producer(),
        )
        cfg2 = KafkaConfig(
            bootstrap_servers="host2:9093",
            security_protocol="SSL",
            client_id_prefix="svc2",
            consumer_config=c2,
            producer_config=self._make_producer(),
        )

        self.assertEqual(cfg1.consumer_config.auto_offset_reset, "earliest")
        self.assertEqual(cfg2.consumer_config.auto_offset_reset, "latest")
        self.assertNotEqual(cfg1.bootstrap_servers, cfg2.bootstrap_servers)

    def test_supports_job_configs_with_event_type(self) -> None:
        job = KafkaJobConfig(
            topic="mlflow-tracking",
            group_id="mlflow-worker",
            event_type="yolo_detector",
            consumer=self._make_consumer(),
        )
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            security_protocol="PLAINTEXT",
            client_id_prefix="test-service",
            consumer_config=self._make_consumer(),
            producer_config=self._make_producer(),
            jobs={"tracking_detection": job},
        )

        self.assertEqual(config.jobs["tracking_detection"].topic, "mlflow-tracking")
        self.assertEqual(config.jobs["tracking_detection"].group_id, "mlflow-worker")
        self.assertEqual(config.jobs["tracking_detection"].event_type, "yolo_detector")


if __name__ == "__main__":
    unittest.main()
