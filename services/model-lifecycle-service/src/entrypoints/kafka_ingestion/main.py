from __future__ import annotations

import asyncio
import signal
import subprocess
import threading
from typing import Any

from confluent_kafka import Consumer as ConfluentConsumer, Message

from src.modules.job_control.adapters.inbound.kafka.ingest_job_handler import (
    IngestJobHandler,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_record_repository import (
    PostgresJobRecordRepository,
)
from src.modules.job_control.application.use_case.ingest_job import (
    IngestJob,
)
from src.platform.config import ConfigLoader
from src.platform.logger import Logger, LoggerConfig
from src.platform.messaging.kafka.config import KafkaConfig, KafkaJobConfig
from src.platform.messaging.kafka.message import ConsumedMessage
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.kafka_event_schema_guard import (
    KafkaEventSchemaGuard,
)
from src.platform.persistence.postgres.transaction import PostgresTransaction


def _build_consumer_config(
    kafka_config: KafkaConfig,
    job_config: KafkaJobConfig,
) -> dict[str, str | int | bool]:
    return {
        "bootstrap.servers": kafka_config.bootstrap_servers,
        "security.protocol": kafka_config.security_protocol,
        "group.id": job_config.group_id,
        "client.id": f"{kafka_config.client_id_prefix}-mlflow-tracking",
        "auto.offset.reset": kafka_config.consumer_config.auto_offset_reset,
        "enable.auto.commit": False,
        "session.timeout.ms": kafka_config.consumer_config.session_timeout_ms,
        "max.poll.interval.ms": kafka_config.consumer_config.max_poll_interval_ms,
    }


def _run_ingestion_consumer(
    kafka_config: KafkaConfig,
    job_config: KafkaJobConfig,
    pg_dsn: str,
    logger: Logger,
    stop_event: threading.Event,
) -> None:
    event_types = job_config.event_types()
    thread_logger = Logger(f"IngestionConsumer-{job_config.group_id}")
    thread_logger.info(
        f"[INGESTION_CONSUMER_STARTING] event_types={event_types} "
        f"topic={job_config.topic} group={job_config.group_id}"
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool: PostgresPool | None = None
    consumer: ConfluentConsumer | None = None

    try:
        pool = PostgresPool(dsn=pg_dsn, logger=thread_logger)
        loop.run_until_complete(pool.start())
        async def _ensure_indexes() -> None:
            async with pool.acquire() as connection:
                await KafkaEventSchemaGuard(logger=thread_logger).ensure_indexes(
                    connection
                )

        loop.run_until_complete(_ensure_indexes())
        transaction = PostgresTransaction(pool=pool)
        record_repo = PostgresJobRecordRepository(
            transaction=transaction, logger=thread_logger
        )
        ingest_job = IngestJob(
            repository=record_repo,
            logger=thread_logger,
            expected_event_types=event_types,
        )
        handler = IngestJobHandler(
            ingest_job=ingest_job,
            logger=thread_logger,
            consumer_group=job_config.group_id,
        )

        consumer = ConfluentConsumer(_build_consumer_config(kafka_config, job_config))
        consumer.subscribe([job_config.topic])

        thread_logger.info(
            f"[INGESTION_CONSUMER_STARTED] event_types={event_types} "
            f"topic={job_config.topic}"
        )

        while not stop_event.is_set():
            msg: Message | None = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                thread_logger.error(f"[CONSUMER_ERROR] {msg.error()}")
                continue

            consumed = ConsumedMessage(
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
                key=msg.key(),
                value=msg.value(),
                headers=tuple(msg.headers() or ()),
                timestamp_ms=msg.timestamp()[1],
                timestamp_type=msg.timestamp()[0],
            )

            thread_logger.info(
                f"[MESSAGE_RECEIVED] topic={msg.topic()} "
                f"partition={msg.partition()} offset={msg.offset()}"
            )

            try:
                loop.run_until_complete(handler.handle(consumed))
            except Exception as exc:
                thread_logger.exception(
                    f"[INGESTION_FAILED] event_types={event_types} "
                    f"offset={msg.offset()}: {exc}"
                )
                raise

            consumer.commit(message=msg, asynchronous=False)
            thread_logger.info(
                f"[MESSAGE_COMMITTED] topic={msg.topic()} "
                f"offset={msg.offset()}"
            )

    except Exception as exc:
        thread_logger.exception(
            f"[CONSUMER_CRASHED] event_types={event_types}: {exc}"
        )
    finally:
        if consumer is not None:
            consumer.close()
        if pool is not None:
            loop.run_until_complete(pool.stop())
        loop.close()
        thread_logger.info(
            f"[INGESTION_CONSUMER_STOPPED] event_types={event_types}"
        )


def main() -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("KafkaIngestionDaemon")
    logger.info("[KAFKA_INGESTION_DAEMON_BOOTSTRAP]")

    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()
    env_file = f"{pwd}/services/model-lifecycle-service/config/.env"
    config_file = (
        f"{pwd}/services/model-lifecycle-service/config/"
        "model_lifecycle_orchestrator_config.yaml"
    )

    kafka_config = ConfigLoader.load(
        KafkaConfig,
        yaml_files=[config_file],
        env_files=[env_file],
        section={
            "kafka": None,
            "kafka.defaults.consumer": "consumer_config",
            "kafka.defaults.producer": "producer_config",
            "kafka.jobs": "jobs",
        },
    )
    pg_config = ConfigLoader.load(
        PostgresConfig,
        yaml_files=[config_file],
        env_files=[env_file],
        section="PostgresSql",
    )

    job_names = ["mlflow_tracking", "mlflow_download", "export_onnx", "training", "dataset", "continual_learning", "lifecycle"]

    jobs: dict[str, KafkaJobConfig] = {}
    for name in job_names:
        job = kafka_config.jobs.get(name)
        if job is None:
            logger.warning(
                f"[JOB_NOT_FOUND] No '{name}' entry in kafka.jobs config. Skipping."
            )
            continue
        jobs[name] = job

    if not jobs:
        logger.error(
            "[NO_JOBS_CONFIGURED] No valid job entries found in kafka.jobs config. "
            f"Available: {list(kafka_config.jobs.keys())}"
        )
        return

    logger.info(
        f"[LOADED_CONFIG] bootstrap_servers={kafka_config.bootstrap_servers} "
        f"jobs={list(jobs.keys())}"
    )

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info(f"[SHUTDOWN_SIGNAL] signal={signum}")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    threads: list[threading.Thread] = []
    pg_dsn = pg_config.to_dsn()

    for name, job in jobs.items():
        logger.info(
            f"[STARTING_CONSUMER] name={name} topic={job.topic} "
            f"group_id={job.group_id} event_types={job.event_types()}"
        )
        t = threading.Thread(
            target=_run_ingestion_consumer,
            args=(kafka_config, job, pg_dsn, logger, stop_event),
            daemon=True,
            name=f"ingestion-{name}",
        )
        t.start()
        threads.append(t)

    logger.info("[KAFKA_INGESTION_DAEMON_STARTED]")

    stop_event.wait()

    for t in threads:
        t.join(timeout=30)

    logger.info("[KAFKA_INGESTION_DAEMON_STOPPED]")


if __name__ == "__main__":
    main()
