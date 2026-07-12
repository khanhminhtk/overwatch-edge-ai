from __future__ import annotations

import asyncio
import math
import socket
from pathlib import Path
from typing import Any

from psycopg import sql

from src.modules.continual_learning.adapters.inbound.job_control.continual_learning_job_handler import (
    ContinualLearningJobHandler,
)
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
from src.platform.config import ConfigLoader
from src.platform.logger import Logger, LoggerConfig
from src.platform.messaging.kafka.config import KafkaConfig
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.kafka_event_schema_guard import (
    KafkaEventSchemaGuard,
)
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.transaction import PostgresTransaction
from src.platform.vision import GoogleVisionConfig, GoogleVisionOCR

SERVICE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "model_lifecycle_orchestrator_config.yaml"

CL_RETURNING_SQL = """
e.id,
e.request_id,
e.payload->>'raw_dir' AS raw_dir,
e.payload->>'output_dir' AS output_dir,
e.payload->>'class_id' AS class_id,
e.status
"""


def _build_cl_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_dir": row.get("raw_dir"),
        "output_dir": row.get("output_dir"),
        "class_id": row.get("class_id", "0"),
    }


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
    handler: ContinualLearningJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> PollingJobRunner:
    claim_repository = MLflowJobRepository(
        transaction=transaction,
        logger=Logger(f"ContinualLearningJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        returning=sql.SQL(CL_RETURNING_SQL),
        payload_builder=_build_cl_payload,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimMlflowTrackingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextContinualLearningJob-{event_type}"),
    )
    return PollingJobRunner(
        process_next_job=process_next_job,
        server_id=server_id,
        event_type=event_type,
        logger=Logger(f"ContinualLearningPollingRunner-{event_type}"),
        idle_sleep_seconds=idle_sleep_seconds,
    )


async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("ContinualLearningDaemon")
    logger.info("[CONTINUAL_LEARNING_DAEMON_BOOTSTRAP]")
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

    job_config = kafka_config.jobs.get("continual_learning")
    if job_config is None:
        logger.error(
            "[JOB_CONFIG_NOT_FOUND] No 'continual_learning' entry in kafka.jobs config. "
            f"Available: {list(kafka_config.jobs.keys())}"
        )
        return

    event_types = job_config.event_types()
    if len(event_types) != 1:
        logger.error(
            "[INVALID_EVENT_TYPES] Expected exactly 1 event_type in "
            f"'continual_learning' config, got: {event_types}"
        )
        return

    event_type = event_types[0]

    vision_config = ConfigLoader.load(
        GoogleVisionConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="GoogleVision",
    )
    vision_config.convert_path_to_absolute(str(SERVICE_ROOT))
    vision_model = GoogleVisionOCR(config=vision_config)

    pool = PostgresPool(dsn=pg_config.to_dsn(), logger=logger)
    await pool.start()
    async with pool.acquire() as connection:
        await KafkaEventSchemaGuard(logger=logger).ensure_indexes(connection)
    transaction = PostgresTransaction(pool=pool)
    record_repository = PostgresJobRecordRepository(
        transaction=transaction,
        logger=logger,
    )
    handler = ContinualLearningJobHandler(
        vision_model=vision_model,
        logger=Logger("ContinualLearningJobHandler"),
    )

    runner = _build_runner(
        event_type=event_type,
        server_id=socket.gethostname(),
        transaction=transaction,
        record_repository=record_repository,
        handler=handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
    )

    logger.info(
        "[CONTINUAL_LEARNING_DAEMON_STARTED]",
        f"event_type={event_type}",
        f"topic={job_config.topic}",
        f"group_id={job_config.group_id}",
    )
    logger.info(
        "[CONTINUAL_LEARNING_DAEMON_MODE]",
        "This daemon claims jobs from kafka_events only. "
        "Run kafka_ingestion.main to ingest Kafka messages into Postgres first.",
    )
    try:
        await runner.run_forever()
    except asyncio.CancelledError:
        logger.info("[CONTINUAL_LEARNING_DAEMON_CANCELLED]")
    finally:
        await pool.stop()
        logger.info("[CONTINUAL_LEARNING_DAEMON_STOPPED]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Continual learning daemon; claim and process continual learning jobs from kafka_events."
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
        default=300,
        help="Seconds after which a PROCESSING job is considered stale and reclaimable (default: 300)",
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
