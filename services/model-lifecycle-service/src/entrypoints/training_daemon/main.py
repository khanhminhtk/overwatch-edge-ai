from __future__ import annotations

import asyncio
import os
import socket

from src.bootstrap import (
    load_job_control_config,
    load_kafka_config,
    load_postgres_config,
    resolve_reclaim_timeout,
    run_runners_until_shutdown,
    start_postgres_runtime,
    validate_runtime_inputs,
    build_success_event_publisher,
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
from src.modules.training.adapters.inbound.job_control.training_job_handler import (
    TrainingJobHandler,
)
from src.modules.training.application.use_case import (
    TrainDetectionUseCase,
    TrainRecognizerUseCase,
)
from src.platform.logger import Logger, LoggerConfig
from src.platform.persistence.postgres.transaction import PostgresTransaction


def _build_runner(
    *,
    event_type: str,
    job_name: str,
    server_id: str,
    transaction: PostgresTransaction,
    record_repository: PostgresJobRecordRepository,
    handler: TrainingJobHandler,
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
    success_event_publisher,
) -> PollingJobRunner:
    claim_repository = PostgresClaimedJobRepository(
        transaction=transaction,
        logger=Logger(f"TrainingJobRepo-{event_type}"),
        event_type=event_type,
        reclaim_timeout_seconds=reclaim_timeout_seconds,
    )
    process_next_job = ProcessNextJob(
        claim_job=ClaimNextPendingJob(repository=claim_repository),
        handler=handler,
        mark_processed=MarkJobProcessed(repository=record_repository),
        mark_failed=MarkJobFailed(repository=record_repository),
        logger=Logger(f"ProcessNextTrainingJob-{event_type}"),
        job_name=job_name,
        success_event_publisher=success_event_publisher,
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
    reclaim_timeout_seconds: int | None,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("TrainingDaemon")
    logger.info("[TRAINING_DAEMON_BOOTSTRAP]")
    validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = load_kafka_config()
    pg_config = load_postgres_config()
    job_control_config = load_job_control_config()

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
    resolved_reclaim_timeout_seconds = resolve_reclaim_timeout(
        job_name="training",
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
        logger=Logger("TrainingJobSuccessEventPublisher"),
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
        job_name="training",
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=training_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
        success_event_publisher=success_event_publisher,
    )
    detection_runner = _build_runner(
        event_type=detection_event_type,
        job_name="training",
        server_id=server_id,
        transaction=transaction,
        record_repository=record_repository,
        handler=training_handler,
        idle_sleep_seconds=idle_sleep_seconds,
        reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
        success_event_publisher=success_event_publisher,
    )

    logger.info(
        "[TRAINING_DAEMON_STARTED]",
        f"recognizer_event_type={recognizer_event_type}",
        f"detection_event_type={detection_event_type}",
        f"default_mode={default_mode}",
        f"reclaim_timeout_seconds={resolved_reclaim_timeout_seconds}",
    )
    try:
        await run_runners_until_shutdown(
            runners=[recognizer_runner, detection_runner],
            logger=logger,
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
