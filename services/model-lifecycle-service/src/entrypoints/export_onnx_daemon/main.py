from __future__ import annotations

import asyncio
import os
import socket

from src.bootstrap import (
    CONFIG_FILE,
    ENV_FILE,
    REPO_ROOT,
    build_success_event_publisher,
    load_job_control_config,
    load_kafka_config,
    load_postgres_config,
    resolve_reclaim_timeout,
    run_runners_until_shutdown,
    start_postgres_runtime,
    validate_runtime_inputs,
)
from src.modules.exports.adapters.inbound.job_control import ExportJobHandler
from src.modules.exports.application.use_case import (
    ExportDetectionUseCase,
    ExportRecognizerUseCase,
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
from src.modules.tracking.domain.value_objects import (
    DetectionConfig,
    RecognizerConfig,
)
from src.platform.config import ConfigLoader
from src.platform.logger import Logger, LoggerConfig
from src.platform.persistence.postgres.transaction import PostgresTransaction


def _build_detection_export(
    detection_config: DetectionConfig,
    logger: Logger,
) -> ExportDetectionUseCase:
    return ExportDetectionUseCase(
        detection_config=detection_config,
        logger=logger,
    )


def _build_recognizer_export(
    recognizer_config: RecognizerConfig,
    logger: Logger,
) -> ExportRecognizerUseCase:
    return ExportRecognizerUseCase(
        recognizer_config=recognizer_config,
        logger=logger,
    )

async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int | None,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("ExportOnnxDaemon")
    logger.info("[EXPORT_ONNX_DAEMON_BOOTSTRAP]")
    validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = load_kafka_config()
    pg_config = load_postgres_config()
    job_control_config = load_job_control_config()
    recognizer_config = ConfigLoader.load(
        RecognizerConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section={
            "mlflow": None,
            "mlflow.recognizer": None,
        },
    )
    detection_config = ConfigLoader.load(
        DetectionConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section={
            "mlflow": None,
            "mlflow.detection": None,
        },
    )

    job_config = kafka_config.jobs.get("export_onnx")
    if job_config is None:
        logger.error(
            "[JOB_CONFIG_NOT_FOUND] No 'export_onnx' entry in "
            f"kafka.jobs config. Available: {list(kafka_config.jobs.keys())}"
        )
        return

    event_types = job_config.event_types()
    if len(event_types) < 2:
        logger.error(
            "[INVALID_EVENT_TYPES] Expected at least 2 event_types in "
            f"'export_onnx' config, got: {event_types}"
        )
        return

    detection_event_type = event_types[0]
    recognizer_event_type = event_types[1]
    resolved_reclaim_timeout_seconds = resolve_reclaim_timeout(
        job_name="export_onnx",
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
        logger=Logger("ExportOnnxJobSuccessEventPublisher"),
    )

    detection_export = _build_detection_export(
        detection_config,
        Logger("ExportDetectionUseCase"),
    )
    recognizer_export = _build_recognizer_export(
        recognizer_config,
        Logger("ExportRecognizerUseCase"),
    )
    export_handler = ExportJobHandler(
        detection_export=detection_export,
        recognizer_export=recognizer_export,
        detection_config=detection_config,
        recognizer_config=recognizer_config,
        detection_event_type=detection_event_type,
        recognizer_event_type=recognizer_event_type,
        project_root=REPO_ROOT,
        onnx_export_dir=os.environ.get("ONNX_EXPORT_DIR", "artifacts/onnx"),
        logger=Logger("ExportJobHandler"),
    )

    server_id = socket.gethostname()

    def _build_runner(
        event_type_value: str,
        name: str,
    ) -> PollingJobRunner:
        claim_repository = PostgresClaimedJobRepository(
            transaction=transaction,
            logger=Logger(f"ExportOnnxJobRepo-{name}"),
            event_type=event_type_value,
            reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
            event_filter="export_onnx",
        )
        process_next_job = ProcessNextJob(
            claim_job=ClaimNextPendingJob(repository=claim_repository),
            handler=export_handler,
            mark_processed=MarkJobProcessed(repository=record_repository),
            mark_failed=MarkJobFailed(repository=record_repository),
            logger=Logger(f"ProcessNextExportJob-{name}"),
            job_name="export_onnx",
            success_event_publisher=success_event_publisher,
        )
        return PollingJobRunner(
            process_next_job=process_next_job,
            server_id=server_id,
            event_type=event_type_value,
            logger=Logger(f"ExportPollingRunner-{name}"),
            idle_sleep_seconds=idle_sleep_seconds,
        )

    detection_runner = _build_runner(detection_event_type, "detection")
    recognizer_runner = _build_runner(recognizer_event_type, "recognizer")

    logger.info(
        "[EXPORT_ONNX_DAEMON_STARTED]",
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
        logger.info("[EXPORT_ONNX_DAEMON_CANCELLED]")
    finally:
        await pool.stop()
        logger.info("[EXPORT_ONNX_DAEMON_STOPPED]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export ONNX daemon; claim and process export jobs from kafka_events."
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
