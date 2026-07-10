from __future__ import annotations

import subprocess

from mlflow import MlflowClient

from src.modules.tracking.application.use_case.detection.artifacts import (
    DetectionMlflowArtifactStore,
)
from src.modules.tracking.application.use_case.detection.registry import (
    DetectionMlflowModelRegistry,
)
from src.modules.tracking.application.use_case.detection.tracking import (
    DetectionMlflowTracking,
)
from src.modules.tracking.application.use_case.detection.workflow import (
    DetectionMlflowWorkflow,
)
from src.modules.tracking.domain.entity_objects import DetectionRun
from src.modules.tracking.domain.value_objects import DetectionConfig
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.tracking.mlflow import (
    MlflowArtifact,
    MlflowRegistry,
    MlflowTrace,
    MlflowTracking,
)
from src.platform.tracking.mlflow.config import MlflowConfig


def main() -> None:
    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()

    detection_config = ConfigLoader.load(
        DetectionConfig,
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        section={
            "mlflow": None,
            "mlflow.detection": None,
        },
    )
    print(f"Loaded DetectionConfig: {detection_config}")

    mlflow_config = ConfigLoader.load(
        MlflowConfig,
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        section="mlflow",
    )

    run = DetectionRun(
        experiment_name="test_experiment",
        run_name="test_run",
        config=detection_config,
    )

    client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
    tracking = MlflowTracking(client=client, tracking_url=mlflow_config.tracking_uri)
    registry = MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri)
    artifact_store = MlflowArtifact(client=client, tracking_url=mlflow_config.tracking_uri)
    trace_port = MlflowTrace(client=client, tracking_url=mlflow_config.tracking_uri)

    tracking_logger = Logger("DetectionMlflowTracking")
    artifact_logger = Logger("DetectionMlflowArtifactStore")
    registry_logger = Logger("DetectionMlflowModelRegistry")

    tracking_step = DetectionMlflowTracking(
        run=run,
        hardware_info=detection_config.hardware,
        experiment_tracker=tracking,
        trace_port=trace_port,
        logger=tracking_logger,
    )
    artifact_step = DetectionMlflowArtifactStore(
        run=run,
        tracker=tracking,
        artifact_store=artifact_store,
        logger=artifact_logger,
    )
    registry_step = DetectionMlflowModelRegistry(
        run=run,
        registry=registry,
        tracker=tracking,
        logger=registry_logger,
    )

    workflow = DetectionMlflowWorkflow(
        tracking=tracking_step,
        artifacts=artifact_step,
        registry=registry_step,
        logger=Logger("DetectionMlflowWorkflow"),
    )
    model_version = workflow.execute(
        checkpoint_names=[
            detection_config.best_checkpoint_name,
            detection_config.last_checkpoint_name,
        ],
        pwd=pwd,
    )
    print(f"Registered detection model version: {model_version}")


if __name__ == "__main__":
    main()
