from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from mlflow import MlflowClient

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
from src.platform.tracking.mlflow import MlflowRegistry
from src.platform.tracking.mlflow.config import MlflowConfig


def _build_detection_download(
    mlflow_config: MlflowConfig,
    detection_config: DetectionConfig,
    logger: Logger,
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
    logger: Logger,
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


async def async_main(download_interval_seconds: float) -> None:
    Logger.configure(LoggerConfig())
    logger = Logger("MLflowDownloadDaemon")
    logger.info("[MLFLOW_DOWNLOAD_DAEMON_BOOTSTRAP]")

    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()
    env_file = f"{pwd}/services/model-lifecycle-service/config/.env"
    config_file = (
        f"{pwd}/services/model-lifecycle-service/config/"
        "model_lifecycle_orchestrator_config.yaml"
    )

    mlflow_config = ConfigLoader.load(
        MlflowConfig,
        yaml_files=[config_file],
        env_files=[env_file],
        section="mlflow",
    )
    recognizer_config = ConfigLoader.load(
        RecognizerConfig,
        yaml_files=[config_file],
        env_files=[env_file],
        section={
            "mlflow": None,
            "mlflow.recognizer": None,
        },
    )
    detection_config = ConfigLoader.load(
        DetectionConfig,
        yaml_files=[config_file],
        env_files=[env_file],
        section={
            "mlflow": None,
            "mlflow.detection": None,
        },
    )

    detection_download = _build_detection_download(mlflow_config, detection_config, logger)
    recognizer_download = _build_recognizer_download(mlflow_config, recognizer_config, logger)

    logger.info("[MLFLOW_DOWNLOAD_DAEMON_STARTED]")

    try:
        while True:
            detection_output = Path(detection_config.checkpoint_dir) / detection_config.best_checkpoint_name
            detection_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                detection_download.download_champion_to_file(
                    artifact_path=detection_config.best_checkpoint_name,
                    output_path=str(detection_output),
                )
                logger.info(
                    "[DETECTION_DOWNLOADED]",
                    f"output={detection_output}",
                )
            except Exception as exc:
                logger.error(
                    "[DETECTION_DOWNLOAD_FAILED]",
                    f"error={exc}",
                )

            recognizer_output = Path(recognizer_config.checkpoint_dir) / recognizer_config.best_checkpoint_name
            recognizer_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                recognizer_download.download_champion_to_file(
                    artifact_path=recognizer_config.best_checkpoint_name,
                    output_path=str(recognizer_output),
                )
                logger.info(
                    "[RECOGNIZER_DOWNLOADED]",
                    f"output={recognizer_output}",
                )
            except Exception as exc:
                logger.error(
                    "[RECOGNIZER_DOWNLOAD_FAILED]",
                    f"error={exc}",
                )

            await asyncio.sleep(download_interval_seconds)
    except asyncio.CancelledError:
        logger.info("[MLFLOW_DOWNLOAD_DAEMON_CANCELLED]")
    finally:
        logger.info("[MLFLOW_DOWNLOAD_DAEMON_STOPPED]")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="MLflow Download Daemon — periodically download champion model artifacts"
    )
    parser.add_argument(
        "--download-interval",
        type=float,
        default=300.0,
        help="Seconds between download cycles (default: 300.0)",
    )
    args = parser.parse_args()

    asyncio.run(
        async_main(
            download_interval_seconds=args.download_interval,
        )
    )


if __name__ == "__main__":
    main()
