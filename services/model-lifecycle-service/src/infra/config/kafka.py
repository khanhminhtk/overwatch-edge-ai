from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KafkaConsumerConfig:
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    session_timeout_ms: int = 45000
    max_poll_interval_ms: int = 300000


@dataclass(frozen=True)
class KafkaProducerConfig:
    acks: str = "all"
    enable_idempotence: bool = True
    retries: int = 3
    linger_ms: int = 5
    retry_backoff_ms: int = 1000
    compression_type: str = "snappy"


@dataclass(frozen=True)
class KafkaDefaultsConfig:
    consumer: KafkaConsumerConfig = field(default_factory=KafkaConsumerConfig)
    producer: KafkaProducerConfig = field(default_factory=KafkaProducerConfig)


@dataclass(frozen=True)
class KafkaJobConfig:
    topic: str = ""
    group_id: str = ""
    consumer: KafkaConsumerConfig = field(default_factory=KafkaConsumerConfig)
    producer: KafkaProducerConfig | None = None


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    client_id_prefix: str = "model-lifecycle-service-server"
    defaults: KafkaDefaultsConfig = field(default_factory=KafkaDefaultsConfig)
    jobs: dict[str, KafkaJobConfig] = field(default_factory=dict)


# Backward-compatible aliases for legacy imports.
KafkaConsumerDefaults = KafkaConsumerConfig
KafkaProducerDefaults = KafkaProducerConfig
KafkaTrainReconizerConfig = KafkaJobConfig
KafkaTrainDetectionConfig = KafkaJobConfig
