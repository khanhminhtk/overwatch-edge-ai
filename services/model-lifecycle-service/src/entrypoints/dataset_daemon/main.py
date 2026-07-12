from __future__ import annotations

import asyncio
import socket
from typing import Any

from psycopg import sql

from src.bootstrap import (
    build_success_event_publisher,
    load_job_control_config,
    load_kafka_config,
    load_postgres_config,
    resolve_reclaim_timeout,
    run_runners_until_shutdown,
    start_postgres_runtime,
    validate_runtime_inputs,
)
from src.modules.dataset.adapters.inbound.job_control.dataset_job_handler import (
    DatasetJobHandler,
)
from src.modules.dataset.application.use_case import (
    CreateGodDatasetDetection,
    CreateGodDatasetRecognizer,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_claimed_job_repository import (
    PostgresClaimedJobRepository,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_record_repository import (
    PostgresJobRecordRepository,
)
from src.modules.job_control.adapters.outbound.scheduler.polling_job_runner import (
    PollingJobRunner,
)
from src.modules.job_control.application.use_case.claim_next_pending_job import (
    ClaimNextPendingJob,
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
from src.platform.logger import Logger, LoggerConfig
from src.platform.persistence.postgres.transaction import PostgresTransaction

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


def _build_runner(
    *,
    event_type: str,
    job_name: str,
    server_id: str,
    transaction: PostgresTransaction,
    record_repository: PostgresJobRecordRepository,
    handler: DatasetJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
    success_event_publisher,
) -> PollingJobRunner:
    claim_repository = PostgresClaimedJobRepository(
        transaction=transaction,
        logger=Logger(f"DatasetJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        returning=sql.SQL(DATASET_RETURNING_SQL),
        payload_builder=_build_dataset_payload,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimNextPendingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextDatasetJob-{event_type}"),
        job_name=job_name,
        success_event_publisher=success_event_publisher,
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
    reclaim_timeout_seconds: int | None,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("DatasetDaemon")
    logger.info("[DATASET_DAEMON_BOOTSTRAP]")
    validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = load_kafka_config()
    pg_config = load_postgres_config()
    job_control_config = load_job_control_config()

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
    resolved_reclaim_timeout_seconds = resolve_reclaim_timeout(
        job_name="dataset",
        cli_reclaim_timeout_seconds=reclaim_timeout_seconds,
        job_control_config=job_control_config,
    )

    pool, transaction, record_repository = await start_postgres_runtime(
        pg_config=pg_config,
        logger=logger,
    )
    success_event_publisher = build_success_event_publisher(
        kafka_config=kafka_config,
        job_control_config=job_control_config,
        logger=Logger("DatasetJobSuccessEventPublisher"),
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
        job_name="dataset",
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=dataset_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
        success_event_publisher=success_event_publisher,
    )
    recognizer_runner = _build_runner(
        event_type=recognizer_event_type,
        job_name="dataset",
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=dataset_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
        success_event_publisher=success_event_publisher,
    )

    logger.info(
        "[DATASET_DAEMON_STARTED]",
        f"detection_event_type={detection_event_type}",
        f"recognizer_event_type={recognizer_event_type}",
        f"reclaim_timeout_seconds={resolved_reclaim_timeout_seconds}",
    )
    try:
        await run_runners_until_shutdown(
            runners=[detection_runner, recognizer_runner],
            logger=logger,
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
        default=None,
        help="Override reclaim timeout seconds for stale PROCESSING jobs (default: from config)",
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
