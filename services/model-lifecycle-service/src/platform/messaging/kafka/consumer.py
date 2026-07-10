import asyncio
import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from threading import Event

from confluent_kafka import Consumer as ConfluentConsumer
from confluent_kafka import KafkaException, KafkaError, Message, TopicPartition

from src.platform.logger import Logger
from src.platform.messaging.kafka.config import KafkaConfig
from src.platform.messaging.kafka.message import KafkaMessageProcessingError, MessageHandler, ConsumedMessage
from src.platform.messaging.kafka.protocols import ConfluentConsumer


class KafkaConsumerClient(ConfluentConsumer):
    def __init__(
        self,
        config: KafkaConfig,
        handler: MessageHandler | Callable[[ConsumedMessage], object],
        group_id: str,
        topics: Sequence[str],
        logger: Logger,
        poll_timeout: float = 1.0
    ):
        if not topics:
            raise ValueError("At least one topic is required")
        self._config = config
        self._handler = handler
        self._poll_timeout = poll_timeout
        self._group_id = group_id
        self._topics = topics
        self._logger = logger
        self._consumer = ConfluentConsumer(self._to_confluent_config(config, group_id))
        print(f"[CONSUMER_INITIALIZED] group={self._group_id} topics={self._topics}")

        self._started = False
        self._closed = False
        self._stop_event = Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_handler(self, handler: MessageHandler | Callable[[ConsumedMessage], object]) -> None:
        if self._started:
            raise RuntimeError("Cannot change handler after consumer started")
        self._handler = handler

    @staticmethod
    def _to_confluent_config(config: KafkaConfig, group_id: str) -> dict[str, str | int | bool]:
        return {
            "bootstrap.servers": config.bootstrap_servers,
            "security.protocol": config.security_protocol,
            "group.id": group_id,
            "client.id": config.client_id_prefix,
            "auto.offset.reset": config.consumer_config.auto_offset_reset,
            "enable.auto.commit": config.consumer_config.enable_auto_commit,
            "session.timeout.ms": config.consumer_config.session_timeout_ms,
            "max.poll.interval.ms": config.consumer_config.max_poll_interval_ms,
        }
    
    def start_consuming(self) -> threading.Thread:
        if self._started:
            raise RuntimeError("Consumer is already running")
        if self._closed:
            raise RuntimeError("Consumer is closed")

        self._thread = threading.Thread(
            target=self.run,
            daemon=True,
            name=f"kafka-consumer-{self._group_id}",
        )
        self._thread.start()
        self._logger.info(
            f"[CONSUMER_THREAD_STARTED] group={self._group_id} "
            f"topics={self._topics} thread={self._thread.name}"
        )
        return self._thread

    def stop(self) -> None:
        if self._stop_event.is_set():
            return

        self._logger.info(f"[CONSUMER_STOP_REQUESTED] group={self._group_id} topics={self._topics}")
        self._stop_event.set()
    
    def run(self) -> None:
        if self._started:
            raise RuntimeError("Consumer is already running")
        if self._closed:
            raise RuntimeError("Consumer is closed")

        is_async = inspect.iscoroutinefunction(self._handler)

        if is_async:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._consumer.subscribe(self._topics)
        self._started = True

        self._logger.info(f"[CONSUMER_STARTED] group={self._group_id} topics={self._topics}")

        try:
            while not self._stop_event.is_set():
                message = self._consumer.poll(timeout=self._poll_timeout)
                if message is None:
                    continue

                if self._has_error(message):
                    self._logger.error(f"[CONSUMER_ERROR] {message.error()}")
                    continue

                print(f"[CONSUMER_MESSAGE_RECEIVED] topic={message.topic()} partition={message.partition()} offset={message.offset()} key={message.key()} value={message.value()}")

                consumed_message = ConsumedMessage(
                    topic=message.topic(),
                    partition=message.partition(),
                    offset=message.offset(),
                    key=message.key(),
                    value=message.value(),
                    headers=tuple(message.headers() or ()),
                    timestamp_ms=message.timestamp()[1],
                    timestamp_type=message.timestamp()[0],
                )
                try:
                    if is_async and self._loop is not None:
                        self._loop.run_until_complete(self._handler(consumed_message))
                    else:
                        self._handler(consumed_message)
                except Exception as e:
                    self._logger.exception(f"[HANDLER_EXCEPTION] {e}")
                    raise KafkaMessageProcessingError(consumed_message) from e
                self._consumer.commit(message=message, asynchronous=False)

        finally:
            if self._loop is not None:
                self._loop.close()
            self._consumer.close()
            self._closed = True
            self._logger.info(f"[CONSUMER_CLOSED] group={self._group_id} topics={self._topics}")

    @staticmethod
    def _has_error(message: Message) -> bool:
        error = message.error()

        if error is None:
            return False

        if error.code() == KafkaError._PARTITION_EOF:
            return True

        raise KafkaException(error)

# if __name__ == "__main__":
#     from src.platform.config import ConfigLoader
#     from src.platform.messaging.kafka.config import KafkaConfig
#     from subprocess import run

#     pwd = run(["pwd"], capture_output=True, text=True).stdout.strip()

#     config_loader = ConfigLoader.load(
#         KafkaConfig,
#         env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
#         yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         section={
#             "kafka": None,
#             "kafka.defaults.consumer": "consumer_config",
#             "kafka.defaults.producer": "producer_config",
#         },
#     )
#     print(config_loader)

#     logger = Logger(name="KafkaConsumerClient")

#     def handle_message(msg: ConsumedMessage) -> None:
#         print(f"[RECEIVED] topic={msg.topic} partition={msg.partition} offset={msg.offset} key={msg.key} value={msg.value}")

#     kafka_consumer_client = KafkaConsumerClient(
#         config=config_loader,
#         handler=handle_message,
#         group_id="test-group",
#         topics=["test-topic"],
#         logger=logger,
#     )

#     try:
#         print("[CONSUMER_RUNNING] Consumer is running. Press Ctrl+C to stop.")
#         kafka_consumer_client.run()
#         print("[CONSUMER_STOPPED] Consumer has been stopped.")
#     except KeyboardInterrupt:
#         print("[CONSUMER_INTERRUPTED] Consumer interrupted by user.")
#         kafka_consumer_client.stop()
