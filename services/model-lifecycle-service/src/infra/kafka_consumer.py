import signal
import time
from types import FrameType
from typing import Callable

try:
    from confluent_kafka import Consumer, KafkaException
except ImportError:  # pragma: no cover
    Consumer = None  # type: ignore[assignment]

    class KafkaException(Exception):
        pass

from src.utils.logger import Logger
from src.infra.config.kafka import KafkaConfig
from src.utils.configloader import ConfigLoader

class GracefulShutdown:
    def __init__(self) -> None:
        self.should_stop = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        print(f"[SHUTDOWN_SIGNAL] signum={signum}")
        self.should_stop = True

class KafkaConsumerClient:
    def __init__(
        self,
        config_loader: ConfigLoader,
        logger: Logger,
        job_key: str | None = None,
        job_keys: list[str] | None = None,
        from_beginning: bool = False,
    ) -> None:
        kafka_config: KafkaConfig = config_loader.get_typed_config(KafkaConfig)
        resolved_job_keys = self._resolve_job_keys(job_key=job_key, job_keys=job_keys)
        job_configs = [kafka_config.jobs[key] for key in resolved_job_keys]
        group_ids = {job_config.group_id for job_config in job_configs}
        if len(group_ids) != 1:
            raise ValueError("All job_keys must share the same consumer group")

        self._topics = [job_config.topic for job_config in job_configs]
        self._logger = logger
        self._from_beginning = from_beginning
        if Consumer is None:
            raise ImportError("confluent_kafka is required to use KafkaConsumerClient")
        self._consumer = Consumer(
            {
                "bootstrap.servers": kafka_config.bootstrap_servers,
                "group.id": job_configs[0].group_id,
                "client.id": kafka_config.client_id_prefix,
                "security.protocol": kafka_config.security_protocol,
                "auto.offset.reset": job_configs[0].consumer.auto_offset_reset,
                "enable.auto.commit": job_configs[0].consumer.enable_auto_commit,
                "session.timeout.ms": job_configs[0].consumer.session_timeout_ms,
                "max.poll.interval.ms": job_configs[0].consumer.max_poll_interval_ms,
            }
        )

    @staticmethod
    def _resolve_job_keys(
        *,
        job_key: str | None,
        job_keys: list[str] | None,
    ) -> list[str]:
        if job_keys:
            return job_keys
        if job_key:
            return [job_key]
        raise ValueError("At least one job_key is required")

    def run(
        self,
        handler: Callable[..., None],
        shutdown: GracefulShutdown,
    ):
        if self._from_beginning:
            self._consumer.subscribe(self._topics, on_assign=self._on_assign_from_beginning)
        else:
            self._consumer.subscribe(self._topics)
        print(f"[CONSUMER_STARTED] topics={self._topics}")

        try:
            while not shutdown.should_stop:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())

                try:
                    handler(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        raw_value=msg.value(),
                    )
                    self._consumer.commit(msg, asynchronous=False)

                    print(
                        "[CONSUMER_COMMIT]",
                        f"topic={msg.topic()}",
                        f"partition={msg.partition()}",
                        f"offset={msg.offset()}",
                    )

                    self._logger.info(
                        "infra.kafka_consumer.KafkaConsumerClient.run Message processed successfully",
                        "[CONSUMER_COMMIT]",
                        f"topic={msg.topic()}",
                        f"partition={msg.partition()}",
                        f"offset={msg.offset()}",
                    )

                except Exception as exc:
                    self._logger.error(
                        "infra.kafka_consumer.KafkaConsumerClient.run Error occurred while processing message",
                        "[CONSUMER_HANDLE_ERROR_NO_COMMIT]",
                        f"error={exc}",
                    )
                    time.sleep(3)

        finally:
            print("[CONSUMER_CLOSING]")
            self._consumer.close()

    def _on_assign_from_beginning(self, consumer, partitions) -> None:
        for partition in partitions:
            partition.offset = 0
        consumer.assign(partitions)
