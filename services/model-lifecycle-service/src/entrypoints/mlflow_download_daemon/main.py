from __future__ import annotations

import asyncio
import socket

from mlflow import MlflowClient

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
from src.modules.tracking.adapters.inbound.job_control import DownloadJobHandler
from src.modules.tracking.application.use_case.detection.download import (
    DetectionMlflowModelDownload,
)
from src.modules.tracking.application.use_case.recognizer.download import (
    RecognizerMlflowModelDownload,
)
from src.modules.tracking.domain.entity_objects import DetectionRun, RecognizerRun
from src.modules.tracking.domain.value_objects import (
    DetectionConfig,
    RecognizerConfig,
)
from src.platform.config import ConfigLoader
from src.platform.logger import Logger, LoggerConfig
from src.platform.persistence.postgres.transaction import PostgresTransaction
from src.platform.tracking.mlflow import MlflowRegistry
from src.platform.tracking.mlflow.config import MlflowConfig


def _build_detection_download(
    mlflow_config: MlflowConfig,
    detection_config: DetectionConfig,
) -> DetectionMlflowModelDownload:
    run = DetectionRun(
        experiment_name=detection_config.model_name,
        run_name=f"{detection_config.model_name}_download",
        config=detection_config,
    )
    client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
    registry = MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri)
    return DetectionMlflowModelDownload(
        run=run,
        registry=registry,
        logger=Logger("DetectionMlflowModelDownload"),
    )


def _build_recognizer_download(
    mlflow_config: MlflowConfig,
    recognizer_config: RecognizerConfig,
) -> RecognizerMlflowModelDownload:
    run = RecognizerRun(
        experiment_name=recognizer_config.model_name,
        run_name=f"{recognizer_config.model_name}_download",
        config=recognizer_config,
    )
    client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
    registry = MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri)
    return RecognizerMlflowModelDownload(
        run=run,
        registry=registry,
        logger=Logger("RecognizerMlflowModelDownload"),
    )


async def async_main(
    idle_sleep_seconds: float,
    reclaim_timeout_seconds: int | None,
    download_timeout_seconds: float,
) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("MLflowDownloadDaemon")
    logger.info("[MLFLOW_DOWNLOAD_DAEMON_BOOTSTRAP]")
    validate_runtime_inputs(idle_sleep_seconds, reclaim_timeout_seconds)

    kafka_config = load_kafka_config()
    pg_config = load_postgres_config()
    job_control_config = load_job_control_config()
    mlflow_config = ConfigLoader.load(
        MlflowConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="mlflow",
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

    job_config = kafka_config.jobs.get("mlflow_download")
    if job_config is None:
        logger.error(
            "[JOB_CONFIG_NOT_FOUND] No 'mlflow_download' entry in "
            f"kafka.jobs config. Available: {list(kafka_config.jobs.keys())}"
        )
        return

    event_types = job_config.event_types()
    if len(event_types) < 2:
        logger.error(
            "[INVALID_EVENT_TYPES] Expected at least 2 event_types in "
            f"'mlflow_download' config, got: {event_types}"
        )
        return

    detection_event_type = event_types[0]
    recognizer_event_type = event_types[1]
    resolved_reclaim_timeout_seconds = resolve_reclaim_timeout(
        job_name="mlflow_download",
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
        logger=Logger("MLflowDownloadJobSuccessEventPublisher"),
    )

    handler = DownloadJobHandler(
        detection_download=_build_detection_download(mlflow_config, detection_config),
        recognizer_download=_build_recognizer_download(mlflow_config, recognizer_config),
        detection_event_type=detection_event_type,
        recognizer_event_type=recognizer_event_type,
        detection_default_checkpoint_name=detection_config.best_checkpoint_name,
        recognizer_default_checkpoint_name=recognizer_config.best_checkpoint_name,
        detection_checkpoint_dir=detection_config.checkpoint_dir,
        recognizer_checkpoint_dir=recognizer_config.checkpoint_dir,
        pwd=str(REPO_ROOT),
        download_timeout_seconds=download_timeout_seconds,
        logger=Logger("DownloadJobHandler"),
    )

    server_id = socket.gethostname()

    def _build_runner(event_type_value: str, name: str) -> PollingJobRunner:
        claim_repository = PostgresClaimedJobRepository(
            transaction=transaction,
            logger=Logger(f"MLflowDownloadJobRepo-{name}"),
            event_type=event_type_value,
            reclaim_timeout_seconds=resolved_reclaim_timeout_seconds,
            event_filter="download_requested",
        )
        process_next_job = ProcessNextJob(
            claim_job=ClaimNextPendingJob(repository=claim_repository),
            handler=handler,
            mark_processed=MarkJobProcessed(repository=record_repository),
            mark_failed=MarkJobFailed(repository=record_repository),
            logger=Logger(f"ProcessNextDownloadJob-{name}"),
            job_name="mlflow_download",
            success_event_publisher=success_event_publisher,
        )
        return PollingJobRunner(
            process_next_job=process_next_job,
            server_id=server_id,
            event_type=event_type_value,
            logger=Logger(f"DownloadPollingRunner-{name}"),
            idle_sleep_seconds=idle_sleep_seconds,
        )

    detection_runner = _build_runner(detection_event_type, "detection")
    recognizer_runner = _build_runner(recognizer_event_type, "recognizer")

    logger.info(
        "[MLFLOW_DOWNLOAD_DAEMON_STARTED]",
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
        logger.info("[MLFLOW_DOWNLOAD_DAEMON_CANCELLED]")
    finally:
        await pool.stop()
        logger.info("[MLFLOW_DOWNLOAD_DAEMON_STOPPED]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MLflow Download Daemon — claim and process download jobs"
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
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=120.0,
        help="Seconds before a single download job is failed as timed out (default: 120.0)",
    )
    args = parser.parse_args()

    asyncio.run(
        async_main(
            idle_sleep_seconds=args.idle_sleep,
            reclaim_timeout_seconds=args.reclaim_timeout_seconds,
            download_timeout_seconds=args.download_timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
