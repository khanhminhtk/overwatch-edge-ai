from __future__ import annotations

import asyncio
import socket
from typing import Any

from psycopg import sql

from src.bootstrap import (
    SERVICE_ROOT,
    build_success_event_publisher,
    load_job_control_config,
    load_kafka_config,
    load_postgres_config,
    resolve_reclaim_timeout,
    run_runners_until_shutdown,
    start_postgres_runtime,
    validate_runtime_inputs,
)
from src.modules.continual_learning.adapters.inbound.job_control.continual_learning_job_handler import (
    ContinualLearningJobHandler,
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
from src.platform.config import ConfigLoader
from src.platform.vision import GoogleVisionConfig, GoogleVisionOCR

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


def _build_runner(
    *,
    event_type: str,
    job_name: str,
    server_id: str,
    transaction: PostgresTransaction,
    record_repository: PostgresJobRecordRepository,
    handler: ContinualLearningJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
    success_event_publisher,
) -> PollingJobRunner:
    claim_repository = PostgresClaimedJobRepository(
        transaction=transaction,
        logger=Logger(f"ContinualLearningJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
        returning=sql.SQL(CL_RETURNING_SQL),
        payload_builder=_build_cl_payload,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimNextPendingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextContinualLearningJob-{event_type}"),
        job_name=job_name,
        success_event_publisher=success_event_publisher,
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
    reclaim_timeout_seconds: int | None,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("ContinualLearningDaemon")
    logger.info("[CONTINUAL_LEARNING_DAEMON_BOOTSTRAP]")
    validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = load_kafka_config()
    pg_config = load_postgres_config()
    job_control_config = load_job_control_config()

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
    resolved_reclaim_timeout_seconds = resolve_reclaim_timeout(
        job_name="continual_learning",
        cli_reclaim_timeout_seconds=reclaim_timeout_seconds,
        job_control_config=job_control_config,
    )

    vision_config = ConfigLoader.load(
        GoogleVisionConfig,
        yaml_files=[SERVICE_ROOT / "config" / "model_lifecycle_orchestrator_config.yaml"],
        env_files=[SERVICE_ROOT / "config" / ".env"],
        section="GoogleVision",
    )
    vision_config.convert_path_to_absolute(str(SERVICE_ROOT))
    vision_model = GoogleVisionOCR(config=vision_config)

    pool, transaction, record_repository = await start_postgres_runtime(
        pg_config=pg_config,
        logger=logger,
    )
    success_event_publisher = build_success_event_publisher(
        kafka_config=kafka_config,
        job_control_config=job_control_config,
        logger=Logger("ContinualLearningJobSuccessEventPublisher"),
    )
    handler = ContinualLearningJobHandler(
        vision_model=vision_model,
        logger=Logger("ContinualLearningJobHandler"),
    )

    runner = _build_runner(
        event_type=event_type,
        job_name="continual_learning",
        server_id=socket.gethostname(),
        transaction=transaction,
        record_repository=record_repository,
        handler=handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
        success_event_publisher=success_event_publisher,
    )

    logger.info(
        "[CONTINUAL_LEARNING_DAEMON_STARTED]",
        f"event_type={event_type}",
        f"topic={job_config.topic}",
        f"group_id={job_config.group_id}",
        f"reclaim_timeout_seconds={resolved_reclaim_timeout_seconds}",
    )
    logger.info(
        "[CONTINUAL_LEARNING_DAEMON_MODE]",
        "This daemon claims jobs from kafka_events only. "
        "Run kafka_ingestion.main to ingest Kafka messages into Postgres first.",
    )
    try:
        await run_runners_until_shutdown(
            runners=[runner],
            logger=logger,
        )
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
