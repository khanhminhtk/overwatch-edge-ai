from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.platform.config import ConfigLoader
from src.platform.messaging.kafka.config import (
    KafkaConfig,
    KafkaConsumerConfig,
    KafkaJobConfig,
    KafkaProducerConfig,
)


class ConfigLoaderDictSectionTest(unittest.TestCase):
    """Test ConfigLoader.load with dict section mapping for KafkaConfig."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return yaml_path

    def test_load_kafka_config_with_dict_section_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                  security_protocol: PLAINTEXT
                  client_id_prefix: test-svc

                  defaults:
                    consumer:
                      auto_offset_reset: earliest
                      enable_auto_commit: false
                      session_timeout_ms: 45000
                      max_poll_interval_ms: 300000
                    producer:
                      acks: all
                      enable_idempotence: true
                      retries: 3
                      linger_ms: 5
                      retry_backoff_ms: 1000
                      compression_type: snappy
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                section={
                    "kafka": None,
                    "kafka.defaults.consumer": "consumer_config",
                    "kafka.defaults.producer": "producer_config",
                },
            )

        self.assertIsInstance(config, KafkaConfig)
        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.security_protocol, "PLAINTEXT")
        self.assertEqual(config.client_id_prefix, "test-svc")

        self.assertIsInstance(config.consumer_config, KafkaConsumerConfig)
        self.assertEqual(config.consumer_config.auto_offset_reset, "earliest")
        self.assertFalse(config.consumer_config.enable_auto_commit)
        self.assertEqual(config.consumer_config.session_timeout_ms, 45000)
        self.assertEqual(config.consumer_config.max_poll_interval_ms, 300000)

        self.assertIsInstance(config.producer_config, KafkaProducerConfig)
        self.assertEqual(config.producer_config.acks, "all")
        self.assertTrue(config.producer_config.enable_idempotence)
        self.assertEqual(config.producer_config.retries, 3)
        self.assertEqual(config.producer_config.linger_ms, 5)
        self.assertEqual(config.producer_config.retry_backoff_ms, 1000)
        self.assertEqual(config.producer_config.compression_type, "snappy")

    def test_load_kafka_config_with_jobs_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                  security_protocol: PLAINTEXT
                  client_id_prefix: test-svc
                  defaults:
                    consumer:
                      auto_offset_reset: earliest
                      enable_auto_commit: false
                      session_timeout_ms: 45000
                      max_poll_interval_ms: 300000
                    producer:
                      acks: all
                      enable_idempotence: true
                      retries: 3
                      linger_ms: 5
                      retry_backoff_ms: 1000
                      compression_type: snappy
                  jobs:
                    tracking_detection:
                      topic: mlflow-tracking
                      group_id: mlflow-worker-detection
                      event_type: yolo_detector
                      consumer:
                        auto_offset_reset: earliest
                        enable_auto_commit: false
                        session_timeout_ms: 45000
                        max_poll_interval_ms: 600000
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                section={
                    "kafka": None,
                    "kafka.defaults.consumer": "consumer_config",
                    "kafka.defaults.producer": "producer_config",
                    "kafka.jobs": "jobs",
                },
            )

        self.assertIsInstance(config.jobs["tracking_detection"], KafkaJobConfig)
        self.assertEqual(config.jobs["tracking_detection"].event_type, "yolo_detector")
        self.assertEqual(config.jobs["tracking_detection"].topic, "mlflow-tracking")

    def test_load_kafka_config_resolves_env_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: ${KAFKA_BOOTSTRAP_SERVERS}
                  security_protocol: ${KAFKA_SECURITY_PROTOCOL}
                  client_id_prefix: test-svc

                  defaults:
                    consumer:
                      auto_offset_reset: ${KAFKA_CONSUMER_AUTO_OFFSET_RESET}
                      enable_auto_commit: false
                      session_timeout_ms: 45000
                      max_poll_interval_ms: 300000
                    producer:
                      acks: all
                      enable_idempotence: true
                      retries: 3
                      linger_ms: 5
                      retry_backoff_ms: 1000
                      compression_type: snappy
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                env={
                    "KAFKA_BOOTSTRAP_SERVERS": "broker1:9092,broker2:9092",
                    "KAFKA_SECURITY_PROTOCOL": "SASL_SSL",
                    "KAFKA_CONSUMER_AUTO_OFFSET_RESET": "latest",
                },
                section={
                    "kafka": None,
                    "kafka.defaults.consumer": "consumer_config",
                    "kafka.defaults.producer": "producer_config",
                },
            )

        self.assertEqual(config.bootstrap_servers, "broker1:9092,broker2:9092")
        self.assertEqual(config.security_protocol, "SASL_SSL")
        self.assertEqual(config.consumer_config.auto_offset_reset, "latest")

    def test_dict_section_none_value_merges_all_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                  security_protocol: PLAINTEXT
                  client_id_prefix: svc

                  defaults:
                    consumer:
                      auto_offset_reset: earliest
                      enable_auto_commit: false
                      session_timeout_ms: 45000
                      max_poll_interval_ms: 300000
                    producer:
                      acks: all
                      enable_idempotence: true
                      retries: 3
                      linger_ms: 5
                      retry_backoff_ms: 1000
                      compression_type: snappy
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                section={
                    "kafka": None,
                    "kafka.defaults.consumer": "consumer_config",
                    "kafka.defaults.producer": "producer_config",
                },
            )

        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.consumer_config.auto_offset_reset, "earliest")
        self.assertEqual(config.producer_config.acks, "all")


class ConfigLoaderStringSectionTest(unittest.TestCase):
    """Test ConfigLoader.load with string section."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return yaml_path

    def test_load_kafka_config_with_string_section_when_yaml_matches_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka_flat:
                  bootstrap_servers: localhost:9092
                  security_protocol: PLAINTEXT
                  client_id_prefix: test-svc
                  consumer_config:
                    auto_offset_reset: earliest
                    enable_auto_commit: false
                    session_timeout_ms: 45000
                    max_poll_interval_ms: 300000
                  producer_config:
                    acks: all
                    enable_idempotence: true
                    retries: 3
                    linger_ms: 5
                    retry_backoff_ms: 1000
                    compression_type: snappy
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                section="kafka_flat",
            )

        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.security_protocol, "PLAINTEXT")
        self.assertEqual(config.client_id_prefix, "test-svc")
        self.assertEqual(config.consumer_config.auto_offset_reset, "earliest")
        self.assertEqual(config.producer_config.acks, "all")


class ConfigLoaderListSectionTest(unittest.TestCase):
    """Test ConfigLoader.load with list section."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return yaml_path

    def test_load_kafka_config_with_list_section_when_yaml_matches_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka_flat:
                  bootstrap_servers: localhost:9092
                  security_protocol: PLAINTEXT
                  client_id_prefix: test-svc
                  consumer_config:
                    auto_offset_reset: earliest
                    enable_auto_commit: false
                    session_timeout_ms: 45000
                    max_poll_interval_ms: 300000
                  producer_config:
                    acks: all
                    enable_idempotence: true
                    retries: 3
                    linger_ms: 5
                    retry_backoff_ms: 1000
                    compression_type: snappy
                """,
            )

            config = ConfigLoader.load(
                KafkaConfig,
                yaml_files=[yaml_path],
                include_os_env=False,
                section=["kafka_flat"],
            )

        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.consumer_config.auto_offset_reset, "earliest")
        self.assertEqual(config.producer_config.acks, "all")


class ConfigLoaderErrorTest(unittest.TestCase):
    """Test ConfigLoader.load error cases for KafkaConfig."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return yaml_path

    def test_dict_section_missing_yaml_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                """,
            )

            with self.assertRaisesRegex(ValueError, r"Missing config section"):
                ConfigLoader.load(
                    KafkaConfig,
                    yaml_files=[yaml_path],
                    include_os_env=False,
                    section={"kafka.nonexistent": "consumer_config"},
                )

    def test_string_section_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                """,
            )

            with self.assertRaisesRegex(ValueError, r"Missing config section"):
                ConfigLoader.load(
                    KafkaConfig,
                    yaml_files=[yaml_path],
                    include_os_env=False,
                    section="nonexistent",
                )

    def test_missing_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = self._write_yaml(
                tmp_path,
                """
                kafka:
                  bootstrap_servers: localhost:9092
                """,
            )

            with self.assertRaisesRegex(ValueError, r"Missing config field"):
                ConfigLoader.load(
                    KafkaConfig,
                    yaml_files=[yaml_path],
                    include_os_env=False,
                    section="kafka",
                )


if __name__ == "__main__":
    unittest.main()
