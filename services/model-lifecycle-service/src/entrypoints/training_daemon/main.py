from __future__ import annotations

import asyncio
import math
import os
import socket
from pathlib import Path

from src.modules.job_control.adapters.outbound.persistence.postgres.mlflow_job_repository import (
    MLflowJobRepository,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_record_repository import (
    PostgresJobRecordRepository,
)
from src.modules.job_control.adapters.outbound.scheduler.polling_job_runner import (
    PollingJobRunner,
)
from src.modules.job_control.application.use_case.claim_mlflow_tracking_job import (
    ClaimMlflowTrackingJob,
)
from src.modules.job_control.application.use_case.mark_job_failed import (
    MarkJobFailed,
)
from src.modules.job_control.application.use_case.mark_job_processed import (
    MarkJobProcessed,
)
from src.modules.job_control.application.use_case.process_next_job import (
    ProcessNextJob,
)
from src.modules.training.adapters.inbound.job_control.training_job_handler import (
    TrainingJobHandler,
)
from src.modules.training.application.use_case import (
    TrainDetectionUseCase,
    TrainRecognizerUseCase,
)
from src.platform.config import ConfigLoader
from src.platform.logger import Logger, LoggerConfig
from src.platform.messaging.kafka.config import KafkaConfig
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.kafka_event_schema_guard import (
    KafkaEventSchemaGuard,
)
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.transaction import PostgresTransaction

SERVICE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "model_lifecycle_orchestrator_config.yaml"


def _validate_runtime_inputs(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> None:
    if not math.isfinite(idle_sleep_seconds) or idle_sleep_seconds <= 0:
        raise ValueError("idle_sleep_seconds must be a finite number greater than 0")
    if reclaim_timeout_seconds <= 0:
        raise ValueError("reclaim_timeout_seconds must be greater than 0")

    missing_paths = [path for path in (CONFIG_FILE, ENV_FILE) if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required daemon config files not found: {missing}")


def _build_runner(
    *,
    event_type: str,
    server_id: str,
    transaction: PostgresTransaction,
    record_repository: PostgresJobRecordRepository,
    handler: TrainingJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
    logger: Logger,
) -> PollingJobRunner:
    claim_repository = MLflowJobRepository(
        transaction=transaction,
        logger=Logger(f"TrainingJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimMlflowTrackingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextTrainingJob-{event_type}"),
    )
    return PollingJobRunner(
        process_next_job=process_next_job,
        server_id=server_id,
        event_type=event_type,
        logger=Logger(f"TrainingPollingRunner-{event_type}"),
        idle_sleep_seconds=idle_sleep_seconds,
    )


async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("TrainingDaemon")
    logger.info("[TRAINING_DAEMON_BOOTSTRAP]")
    _validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = ConfigLoader.load(
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
    pg_config = ConfigLoader.load(
        PostgresConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="PostgresSql",
    )

    job_config = kafka_config.jobs.get("training")
    if job_config is None:
        logger.error(
            "[JOB_CONFIG_NOT_FOUND] No 'training' entry in kafka.jobs config. "
            f"Available: {list(kafka_config.jobs.keys())}"
        )
        return

    event_types = job_config.event_types()
    if len(event_types) < 2:
        logger.error(
            "[INVALID_EVENT_TYPES] Expected at least 2 event_types in "
            f"'training' config, got: {event_types}"
        )
        return

    recognizer_event_type = event_types[0]
    detection_event_type = event_types[1]
    default_mode = os.getenv("TRAINING_MODE", "local")

    pool = PostgresPool(dsn=pg_config.to_dsn(), logger=logger)
    await pool.start()
    async with pool.acquire() as connection:
        await KafkaEventSchemaGuard(logger=logger).ensure_indexes(connection)
    transaction = PostgresTransaction(pool=pool)
    record_repository = PostgresJobRecordRepository(
        transaction=transaction,
        logger=logger,
    )

    recognizer_training = TrainRecognizerUseCase(
        logger=Logger("TrainRecognizerUseCase"),
    )
    detection_training = TrainDetectionUseCase(
        logger=Logger("TrainDetectionUseCase"),
    )
    training_handler = TrainingJobHandler(
        recognizer_training=recognizer_training,
        detection_training=detection_training,
        recognizer_event_type=recognizer_event_type,
        detection_event_type=detection_event_type,
        default_mode=default_mode,
        logger=Logger("TrainingJobHandler"),
    )

    server_id = socket.gethostname()

    recognizer_runner = _build_runner(
        event_type=recognizer_event_type,
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=training_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        logger=logger,
    )
    detection_runner = _build_runner(
        event_type=detection_event_type,
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=training_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        logger=logger,
    )

    logger.info(
        "[TRAINING_DAEMON_STARTED]",
        f"recognizer_event_type={recognizer_event_type}",
        f"detection_event_type={detection_event_type}",
        f"default_mode={default_mode}",
    )
    try:
        await asyncio.gather(
            recognizer_runner.run_forever(),
            detection_runner.run_forever(),
        )
    except asyncio.CancelledError:
        logger.info("[TRAINING_DAEMON_CANCELLED]")
    finally:
        await pool.stop()
        logger.info("[TRAINING_DAEMON_STOPPED]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Training daemon; claim and process training jobs from kafka_events."
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between idle poll cycles (default: 2.0)",
    )
    parser.add_argument(
        "--reclaim-timeout-seconds",
        type=int,
        default=1800,
        help="Seconds after which a PROCESSING job is considered stale and reclaimable (default: 1800)",
    )
    args = parser.parse_args()

    asyncio.run(
        async_main(
            idle_sleep_seconds=args.idle_sleep,
            reclaim_timeout_seconds=args.reclaim_timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
