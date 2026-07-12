from __future__ import annotations

import asyncio
import math
import signal
from dataclasses import dataclass
from pathlib import Path

from src.modules.job_control.adapters.outbound.messaging.kafka.job_success_event_publisher import (
    JobSuccessEventPublisher,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_record_repository import (
    PostgresJobRecordRepository,
)
from src.modules.job_control.adapters.outbound.scheduler.polling_job_runner import (
    PollingJobRunner,
)
from src.modules.job_control.config import JobControlConfig
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.messaging.kafka import KafkaConfig, KafkaProducerClient
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.kafka_event_schema_guard import (
    KafkaEventSchemaGuard,
)
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.transaction import PostgresTransaction


SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVICE_ROOT.parents[1]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "model_lifecycle_orchestrator_config.yaml"


@dataclass(frozen=True)
class WorkerRuntimeContainer:
    kafka_config: KafkaConfig
    pg_config: PostgresConfig
    job_control_config: JobControlConfig


def validate_runtime_inputs(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int | None,
) -> None:
    if not math.isfinite(idle_sleep_seconds) or idle_sleep_seconds <= 0:
        raise ValueError("idle_sleep_seconds must be a finite number greater than 0")
    if reclaim_timeout_seconds is not None and reclaim_timeout_seconds <= 0:
        raise ValueError("reclaim_timeout_seconds must be greater than 0")

    missing_paths = [path for path in (CONFIG_FILE, ENV_FILE) if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required daemon config files not found: {missing}")


def load_kafka_config() -> KafkaConfig:
    return ConfigLoader.load(
        KafkaConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section={
            "kafka": None,
            "kafka.defaults.consumer": "consumer_config",
            "kafka.defaults.producer": "producer_config",
            "kafka.jobs": "jobs",
        },
    )


def load_postgres_config() -> PostgresConfig:
    return ConfigLoader.load(
        PostgresConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="PostgresSql",
    )


def load_job_control_config() -> JobControlConfig:
    return ConfigLoader.load(
        JobControlConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="job_control",
    )


def load_worker_runtime_container() -> WorkerRuntimeContainer:
    return WorkerRuntimeContainer(
        kafka_config=load_kafka_config(),
        pg_config=load_postgres_config(),
        job_control_config=load_job_control_config(),
    )


def resolve_reclaim_timeout(
    *,
    job_name: str,
    cli_reclaim_timeout_seconds: int | None,
    job_control_config: JobControlConfig,
) -> int:
    if cli_reclaim_timeout_seconds is not None:
        return cli_reclaim_timeout_seconds
    return job_control_config.reclaim.timeout_for(job_name)


async def start_postgres_runtime(
    *,
    pg_config: PostgresConfig,
    logger: Logger,
) -> tuple[PostgresPool, PostgresTransaction, PostgresJobRecordRepository]:
    pool = PostgresPool(dsn=pg_config.to_dsn(), logger=logger)
    await pool.start()
    async with pool.acquire() as connection:
        await KafkaEventSchemaGuard(logger=logger).ensure_indexes(connection)
    transaction = PostgresTransaction(pool=pool)
    record_repository = PostgresJobRecordRepository(
        transaction=transaction,
        logger=logger,
    )
    return pool, transaction, record_repository


def build_success_event_publisher(
    *,
    kafka_config: KafkaConfig,
    job_control_config: JobControlConfig,
    logger: Logger,
) -> JobSuccessEventPublisher:
    producer = KafkaProducerClient(
        config=kafka_config,
        logger=Logger("JobSuccessEventProducer"),
    )
    return JobSuccessEventPublisher(
        producer=producer,
        topic=job_control_config.result_topic,
        logger=logger,
    )


async def run_runners_until_shutdown(
    *,
    runners: list[PollingJobRunner],
    logger: Logger,
) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(signal_name: str) -> None:
        logger.info("[WORKER_SHUTDOWN_SIGNAL]", f"signal={signal_name}")
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, _request_stop, signum.name)
        except NotImplementedError:
            pass

    tasks = [asyncio.create_task(runner.run_forever()) for runner in runners]
    stop_waiter = asyncio.create_task(stop_event.wait())
    try:
        done, _pending = await asyncio.wait(
            [*tasks, stop_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_waiter in done:
            for runner in runners:
                await runner.shutdown(reason="worker shutdown signal")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return

        for runner in runners:
            await runner.shutdown(reason="worker runner completed")
        for task in tasks:
            if task in done:
                exc = task.exception()
                if exc is not None:
                    for pending_task in tasks:
                        if pending_task not in done:
                            pending_task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise exc
        for task in tasks:
            if task not in done:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for runner in runners:
            await runner.shutdown(reason="worker cancelled")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        stop_waiter.cancel()
