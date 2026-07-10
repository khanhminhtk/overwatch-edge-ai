from __future__ import annotations

import asyncio
import math
import socket
from pathlib import Path

from src.modules.exports.adapters.inbound.job_control import ExportJobHandler
from src.modules.exports.application.use_case import (
    ExportDetectionUseCase,
    ExportRecognizerUseCase,
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
from src.modules.tracking.domain.value_objects import (
    DetectionConfig,
    RecognizerConfig,
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
REPO_ROOT = SERVICE_ROOT.parents[1]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "model_lifecycle_orchestrator_config.yaml"


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


async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("ExportOnnxDaemon")
    logger.info("[EXPORT_ONNX_DAEMON_BOOTSTRAP]")
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

    pool = PostgresPool(dsn=pg_config.to_dsn(), logger=logger)
    await pool.start()
    async with pool.acquire() as connection:
        await KafkaEventSchemaGuard(logger=logger).ensure_indexes(connection)
    transaction = PostgresTransaction(pool=pool)
    record_repository = PostgresJobRecordRepository(
        transaction=transaction,
        logger=logger,
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
        logger=Logger("ExportJobHandler"),
    )

    server_id = socket.gethostname()

    def _build_runner(
        event_type_value: str,
        name: str,
    ) -> PollingJobRunner:
        claim_repository = MLflowJobRepository(
            transaction=transaction,
            logger=Logger(f"ExportOnnxJobRepo-{name}"),
            event_type=event_type_value,
            reclaim_timeout_seconds=reclaim_timeout_seconds,
            event_filter="export_onnx",
        )
        process_next_job = ProcessNextJob(
            claim_job=ClaimMlflowTrackingJob(repository=claim_repository),
            handler=export_handler,
            mark_processed=MarkJobProcessed(repository=record_repository),
            mark_failed=MarkJobFailed(repository=record_repository),
            logger=Logger(f"ProcessNextExportJob-{name}"),
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
    )
    try:
        await asyncio.gather(
            detection_runner.run_forever(),
            recognizer_runner.run_forever(),
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
