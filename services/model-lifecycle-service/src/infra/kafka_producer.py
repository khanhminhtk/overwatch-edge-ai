import json

try:
    from confluent_kafka import Producer
except ImportError:  # pragma: no cover
    Producer = None  # type: ignore[assignment]

from src.utils.logger import Logger
from src.infra.config.kafka import KafkaConfig
from src.utils.configloader import ConfigLoader

class KafkaProducerClient:
    def __init__(
            self, 
            logger: Logger,
            config_loader: ConfigLoader,
        ) -> None:
        self._logger = logger
        kafka_config = config_loader.get_typed_config(KafkaConfig)
        if Producer is None:
            raise ImportError("confluent_kafka is required to use KafkaProducerClient")
        self._producer = None
        self._producer_config = {
            "bootstrap.servers": kafka_config.bootstrap_servers,
            "client.id": kafka_config.client_id_prefix,
            "security.protocol": kafka_config.security_protocol,
            "acks": kafka_config.defaults.producer.acks,
            "retries": kafka_config.defaults.producer.retries,
            "retry.backoff.ms": kafka_config.defaults.producer.retry_backoff_ms,
            "enable.idempotence": kafka_config.defaults.producer.enable_idempotence,
            "linger.ms": kafka_config.defaults.producer.linger_ms,
            "compression.type": kafka_config.defaults.producer.compression_type,
        }

    def publish(self, topic: str, key: str, value: str, headers: dict = None):
        payload = json.dumps(value).encode("utf-8")
        producer = self._get_or_create_producer()
        producer.produce(
            topic=topic, 
            key=key.encode("utf-8"), 
            value=payload, 
            headers=headers,
            on_delivery=self._on_delivery
        )
        producer.flush()

    def _get_or_create_producer(self):
        if self._producer is None:
            self._producer = Producer(self._producer_config)
        return self._producer

    def _on_delivery(self, err, msg):
        if err is not None:
            self._logger.error(f"[KAFKA_PRODUCER_ERROR] delivery failed: {err}")
            return

        self._logger.info(
            "[KAFKA_PRODUCER_OK]",
            f"topic={msg.topic()}",
            f"partition={msg.partition()}",
            f"offset={msg.offset()}",
        )
