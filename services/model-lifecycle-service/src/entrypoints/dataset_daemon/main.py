from __future__ import annotations

import asyncio
import math
import socket
from pathlib import Path
from typing import Any

from psycopg import sql

from src.modules.dataset.adapters.inbound.job_control.dataset_job_handler import (
    DatasetJobHandler,
)
from src.modules.dataset.application.use_case import (
    CreateGodDatasetDetection,
    CreateGodDatasetRecognizer,
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

SERVICE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "model_lifecycle_orchestrator_config.yaml"

DATASET_RETURNING_SQL = """
e.id,
e.request_id,
e.payload->>'source_data_path' AS source_data_path,
e.payload->>'subset_percent' AS subset_percent,
e.payload->>'output_root' AS output_root,
e.payload->>'dataset_version' AS dataset_version,
e.status
"""


def _build_dataset_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_data_path": row.get("source_data_path", ""),
        "subset_percent": row.get("subset_percent", "10.0"),
        "output_root": row.get("output_root"),
        "dataset_version": row.get("dataset_version", "v1"),
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
    handler: DatasetJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
    logger: Logger,
) -> PollingJobRunner:
    claim_repository = MLflowJobRepository(
        transaction=transaction,
        logger=Logger(f"DatasetJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        returning=sql.SQL(DATASET_RETURNING_SQL),
        payload_builder=_build_dataset_payload,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimMlflowTrackingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextDatasetJob-{event_type}"),
    )
    return PollingJobRunner(
        process_next_job=process_next_job,
        server_id=server_id,
        event_type=event_type,
        logger=Logger(f"DatasetPollingRunner-{event_type}"),
        idle_sleep_seconds=idle_sleep_seconds,
    )


async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("DatasetDaemon")
    logger.info("[DATASET_DAEMON_BOOTSTRAP]")
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

    job_config = kafka_config.jobs.get("dataset")
    if job_config is None:
        logger.error(
            "[JOB_CONFIG_NOT_FOUND] No 'dataset' entry in kafka.jobs config. "
            f"Available: {list(kafka_config.jobs.keys())}"
        )
        return

    event_types = job_config.event_types()
    if len(event_types) < 2:
        logger.error(
            "[INVALID_EVENT_TYPES] Expected at least 2 event_types in "
            f"'dataset' config, got: {event_types}"
        )
        return

    detection_event_type = event_types[0]
    recognizer_event_type = event_types[1]

    pool = PostgresPool(dsn=pg_config.to_dsn(), logger=logger)
    await pool.start()
    async with pool.acquire() as connection:
        await KafkaEventSchemaGuard(logger=logger).ensure_indexes(connection)
    transaction = PostgresTransaction(pool=pool)
    record_repository = PostgresJobRecordRepository(
        transaction=transaction,
        logger=logger,
    )

    detection_dataset = CreateGodDatasetDetection(
        logger=Logger("CreateGodDatasetDetection"),
    )
    recognizer_dataset = CreateGodDatasetRecognizer(
        logger=Logger("CreateGodDatasetRecognizer"),
    )
    dataset_handler = DatasetJobHandler(
        detection_dataset=detection_dataset,
        recognizer_dataset=recognizer_dataset,
        detection_event_type=detection_event_type,
        recognizer_event_type=recognizer_event_type,
        logger=Logger("DatasetJobHandler"),
    )

    server_id = socket.gethostname()

    detection_runner = _build_runner(
        event_type=detection_event_type,
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=dataset_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        logger=logger,
    )
    recognizer_runner = _build_runner(
        event_type=recognizer_event_type,
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=dataset_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        logger=logger,
    )

    logger.info(
        "[DATASET_DAEMON_STARTED]",
        f"detection_event_type={detection_event_type}",
        f"recognizer_event_type={recognizer_event_type}",
    )
    try:
        await asyncio.gather(
            detection_runner.run_forever(),
            recognizer_runner.run_forever(),
        )
    except asyncio.CancelledError:
        logger.info("[DATASET_DAEMON_CANCELLED]")
    finally:
        await pool.stop()
        logger.info("[DATASET_DAEMON_STOPPED]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset daemon; claim and process god-dataset jobs from kafka_events."
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
