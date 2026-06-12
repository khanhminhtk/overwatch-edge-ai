from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from src.infra.config.grpc import GrpcServerConfig, ObjectStorageRuntimeConfig
from src.infra.config.kafka import KafkaConfig, KafkaConsumerConfig, KafkaJobConfig
from src.infra.config.postgrest import PostgrestConfig
from src.utils.configloader import ConfigLoader, load_config


@dataclass(frozen=True)
class NestedItemConfig:
    code: str
    enabled: bool


@dataclass(frozen=True)
class NestedGroupConfig:
    name: str
    items: list[NestedItemConfig]


@dataclass(frozen=True)
class CredentialsConfig:
    username: str
    password: str


@dataclass(frozen=True)
class ExampleConfig:
    grpc_port: int
    kafka_enabled: bool
    threshold: float
    credentials: CredentialsConfig
    groups: dict[str, NestedGroupConfig]
    aliases: tuple[str, str]
    optional_note: Optional[str] = None


@dataclass(frozen=True)
class KafkaOnlyAppConfig:
    kafka: KafkaConfig


class ConfigLoaderTest(unittest.TestCase):
    def test_get_config_resolves_env_variables_in_nested_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_path = tmp_path / ".env"
            yaml_path = tmp_path / "config.yaml"

            env_path.write_text("NESTED_PORT=50051\n", encoding="utf-8")
            yaml_path.write_text(
                textwrap.dedent(
                    """
                    grpc:
                      port: ${NESTED_PORT}
                    """
                ).strip(),
                encoding="utf-8",
            )

            loader = ConfigLoader(env_file=str(env_path), config_yaml_file=str(yaml_path))

            config = loader.get_config()

            self.assertEqual(config["grpc"]["port"], "50051")

    def test_load_config_maps_yaml_to_nested_dataclass_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_path = tmp_path / ".env"
            yaml_path = tmp_path / "config.yaml"

            env_path.write_text(
                "\n".join(
                    [
                        "GRPC_PORT=50051",
                        "KAFKA_ENABLED=false",
                        "THRESHOLD=5.5",
                        "API_PASSWORD=super-secret",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            yaml_path.write_text(
                textwrap.dedent(
                    """
                    grpc_port: ${GRPC_PORT}
                    kafka_enabled: ${KAFKA_ENABLED}
                    threshold: ${THRESHOLD}
                    credentials:
                      username: admin
                      password: ${API_PASSWORD}
                    groups:
                      primary:
                        name: ingest
                        items:
                          - code: alpha
                            enabled: true
                          - code: beta
                            enabled: false
                    aliases:
                      - model-a
                      - model-b
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_config(
                    config_path=yaml_path,
                    dto_cls=ExampleConfig,
                    env_file=env_path,
                )

            self.assertEqual(config.grpc_port, 50051)
            self.assertIs(config.kafka_enabled, False)
            self.assertEqual(config.threshold, 5.5)
            self.assertEqual(config.credentials.password, "super-secret")
            self.assertEqual(config.groups["primary"].items[0].code, "alpha")
            self.assertIs(config.groups["primary"].items[1].enabled, False)
            self.assertEqual(config.aliases, ("model-a", "model-b"))
            self.assertIsNone(config.optional_note)

    def test_load_config_raises_clear_error_for_missing_env_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = tmp_path / "config.yaml"

            yaml_path.write_text(
                textwrap.dedent(
                    """
                    grpc_port: ${GRPC_PORT}
                    kafka_enabled: true
                    threshold: 1.0
                    credentials:
                      username: admin
                      password: secret
                    groups: {}
                    aliases: [a, b]
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    r"Missing environment variable: GRPC_PORT at grpc_port",
                ):
                    load_config(config_path=yaml_path, dto_cls=ExampleConfig)

    def test_load_config_raises_clear_error_for_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = tmp_path / "config.yaml"

            yaml_path.write_text(
                textwrap.dedent(
                    """
                    grpc_port: 50051
                    kafka_enabled: true
                    threshold: 1.0
                    credentials:
                      username: admin
                      password: secret
                    aliases: [a, b]
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"Missing config field: groups",
            ):
                load_config(config_path=yaml_path, dto_cls=ExampleConfig)

    def test_load_config_raises_clear_error_for_invalid_type_cast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yaml_path = tmp_path / "config.yaml"

            yaml_path.write_text(
                textwrap.dedent(
                    """
                    grpc_port: abc
                    kafka_enabled: true
                    threshold: 1.0
                    credentials:
                      username: admin
                      password: secret
                    groups: {}
                    aliases: [a, b]
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r'Invalid config type at grpc_port: expected int, got "abc"',
            ):
                load_config(config_path=yaml_path, dto_cls=ExampleConfig)

    def test_load_config_maps_real_kafka_yaml_to_generic_jobs_dict(self) -> None:
        service_dir = Path(__file__).resolve().parents[1]
        env_path = service_dir / "config" / ".env"
        yaml_path = service_dir / "config" / "model_lifecycle_orchestrator_config.yaml"

        env_values = {
            "kafka_bootstrap_servers": "localhost:9092",
            "kafka_security_protocol": "PLAINTEXT",
            "kafka_client_id_prefix": "model-lifecycle-service-server",
            "kafka_consumer_auto_offset_reset": "earliest",
            "kafka_consumer_enable_auto_commit": "false",
            "kafka_consumer_session_timeout_ms": "45000",
            "kafka_consumer_max_poll_interval_ms": "300000",
            "kafka_producer_acks": "all",
            "kafka_producer_enable_idempotence": "true",
            "kafka_producer_retries": "10",
            "kafka_producer_linger_ms": "5",
            "kafka_producer_retry_backoff_ms": "1000",
            "kafka_producer_compression_type": "snappy",
            "kafka_jobs_train_recognizer_topic": "model.lifecycle.train.recognizer",
            "kafka_jobs_train_recognizer_group_id": "model-lifecycle-train-recognizer",
            "kafka_jobs_train_recognizer_consumer_max_poll_interval_ms": "300000",
            "kafka_jobs_train_detection_topic": "model.lifecycle.train.detection",
            "kafka_jobs_train_detection_group_id": "model-lifecycle-train-detection",
            "kafka_jobs_train_detection_consumer_max_poll_interval_ms": "300000",
        }

        with patch.dict(os.environ, env_values, clear=True):
            config = load_config(
                config_path=yaml_path,
                dto_cls=KafkaOnlyAppConfig,
                env_file=env_path,
            )

        self.assertEqual(config.kafka.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.kafka.defaults.consumer.max_poll_interval_ms, 300000)
        self.assertEqual(config.kafka.defaults.producer.retries, 10)
        self.assertIn("train_recognizer", config.kafka.jobs)
        self.assertIsInstance(config.kafka.jobs["train_recognizer"], KafkaJobConfig)
        self.assertIsInstance(
            config.kafka.jobs["train_recognizer"].consumer,
            KafkaConsumerConfig,
        )
        self.assertEqual(
            config.kafka.jobs["train_recognizer"].topic,
            "model.lifecycle.train.recognizer",
        )
        self.assertEqual(
            config.kafka.jobs["train_detection"].group_id,
            "model-lifecycle-train-detection",
        )

    def test_load_config_can_select_nested_kafka_section_directly(self) -> None:
        service_dir = Path(__file__).resolve().parents[1]
        env_path = service_dir / "config" / ".env"
        yaml_path = service_dir / "config" / "model_lifecycle_orchestrator_config.yaml"

        config = load_config(
            config_path=yaml_path,
            dto_cls=KafkaConfig,
            env_file=env_path,
        )

        self.assertEqual(config.bootstrap_servers, "localhost:9092")
        self.assertEqual(config.security_protocol, "PLAINTEXT")
        self.assertIn("train_recognizer", config.jobs)
        self.assertEqual(
            config.jobs["train_recognizer"].group_id,
            "model-lifecycle-train-recognizer",
        )

    def test_load_config_can_map_explicit_postgres_section(self) -> None:
        service_dir = Path(__file__).resolve().parents[1]
        env_path = service_dir / "config" / ".env"
        yaml_path = service_dir / "config" / "model_lifecycle_orchestrator_config.yaml"

        config = load_config(
            config_path=yaml_path,
            dto_cls=PostgrestConfig,
            env_file=env_path,
            config_section="PostgresSql",
        )

        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 5432)
        self.assertEqual(config.database, "overwatch")
        self.assertEqual(config.user, "postgres")
        self.assertEqual(config.password, "postgres")

    def test_load_config_maps_grpc_server_config_from_root_yaml(self) -> None:
        service_dir = Path(__file__).resolve().parents[1]
        env_path = service_dir / "config" / ".env"
        yaml_path = service_dir / "config" / "model_lifecycle_orchestrator_config.yaml"

        config = load_config(
            config_path=yaml_path,
            dto_cls=GrpcServerConfig,
            env_file=env_path,
        )

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 50051)

    def test_load_config_maps_object_storage_runtime_config_from_root_yaml(self) -> None:
        service_dir = Path(__file__).resolve().parents[1]
        env_path = service_dir / "config" / ".env"
        yaml_path = service_dir / "config" / "model_lifecycle_orchestrator_config.yaml"

        config = load_config(
            config_path=yaml_path,
            dto_cls=ObjectStorageRuntimeConfig,
            env_file=env_path,
        )

        self.assertEqual(config.grpc_host, "localhost")
        self.assertEqual(config.grpc_port, 50001)
        self.assertEqual(config.refresh_token[:8], "40000a39")
        self.assertEqual(config.data_raw_recognizer.split("/")[-1], "recognizer")


if __name__ == "__main__":
    unittest.main()
