from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.workflow import BaseMlflowWorkflowUseCase
from src.applications.use_cases.mlflow.detection.mlflow_artifacts import MlflowArtifactsDetection
from src.applications.use_cases.mlflow.detection.mlflow_model_registry import (
    MlflowModelRegistryDetection,
)
from src.applications.use_cases.mlflow.detection.mlflow_tracking import MlflowTrackingDetection
from src.infra.mlflow.mlflow_artifact import MlflowArtifact
from src.infra.mlflow.mlflow_registry import MlflowRegistry
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.configloader import load_environment
from src.utils.logger import Logger


class MLflowDetection(BaseMlflowWorkflowUseCase):
    def __init__(
        self,
        mlflow_tracking_detection: MlflowTrackingDetection,
        mlflow_artifacts: MlflowArtifactsDetection,
        mlflow_model_registry: MlflowModelRegistryDetection,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
    ):
        super().__init__(
            mlflow_tracking=mlflow_tracking,
            artifacts_use_case=mlflow_artifacts,
            tracking_use_case=mlflow_tracking_detection,
            model_registry_use_case=mlflow_model_registry,
            logger=logger,
            experiment_name=os.getenv("DETECT_MODEL_TYPE", "detector"),
            run_name_prefix="detector",
            model_name=os.getenv("MLFLOW_MODEL_NAME_DETECTION", "yolo_detector"),
            gpu_name=os.getenv("GPU_NAME", "unknown_gpu"),
        )


if __name__ == "__main__":
    tracking_uri = "http://localhost:5000"
    env_path = "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/config/.env.example"
    load_environment(env_path)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    pwd = "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus"

    logger = Logger(__name__)
    mlflow_tracking = MlflowTracking(tracking_uri=tracking_uri)
    mlflow_registry = MlflowRegistry(tracking_uri=tracking_uri)
    mlflow_artifact = MlflowArtifact(mlflow_tracking.client)
    mlflow_tracking_detection = MlflowTrackingDetection(
        mlflow_tracking=mlflow_tracking,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    mlflow_artifacts = MlflowArtifactsDetection(
        mlflow_tracking=mlflow_tracking,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    mlflow_model_registry = MlflowModelRegistryDetection(
        mlflow_model_registry=mlflow_registry,
        mlflow_artifact=mlflow_artifact,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    detection = MLflowDetection(
        mlflow_tracking_detection=mlflow_tracking_detection,
        mlflow_artifacts=mlflow_artifacts,
        mlflow_model_registry=mlflow_model_registry,
        mlflow_tracking=mlflow_tracking,
        logger=logger,
    )
    detection.execute(
        git_commit=git_commit,
        checkpoint_name=[
            os.getenv("DETECT_BEST_CHECKPOINT_NAME", "best.pt"),
            os.getenv("DETECT_LAST_CHECKPOINT_NAME", "last.pt"),
        ],
        goal=os.getenv("MLFLOW_RUN_GOAL_DETECTION", "training"),
        tracking_level=os.getenv("MLFLOW_TRACKING_LEVEL_DETECTION", "full"),
        pwd=pwd,
    )
