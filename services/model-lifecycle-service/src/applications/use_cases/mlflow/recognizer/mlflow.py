import os
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.workflow import BaseMlflowWorkflowUseCase
from src.applications.use_cases.mlflow.recognizer.mlflow_tracking import MlflowTrackingRecognizer
from src.applications.use_cases.mlflow.recognizer.mlflow_artifacts import MlflowArtifactsRecognizer
from src.applications.use_cases.mlflow.recognizer.mlflow_model_registry import MlflowModelRegistryRecognizer
from src.infra.mlflow.mlflow_artifact import MlflowArtifact
from src.infra.mlflow.mlflow_registry import MlflowRegistry
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.configloader import load_environment
from src.utils.logger import Logger

class MLflowRecognizer(BaseMlflowWorkflowUseCase):
    def __init__(
        self,
        mlflow_tracking_recognizer: MlflowTrackingRecognizer,
        mlflow_artifacts: MlflowArtifactsRecognizer,
        mlflow_model_registry: MlflowModelRegistryRecognizer,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
    ):
        super().__init__(
            mlflow_tracking=mlflow_tracking,
            artifacts_use_case=mlflow_artifacts,
            tracking_use_case=mlflow_tracking_recognizer,
            model_registry_use_case=mlflow_model_registry,
            logger=logger,
            experiment_name=os.getenv("RECOG_MODEL_TYPE", "recognizer"),
            run_name_prefix="recognizer",
            model_name=os.getenv("MLFLOW_MODEL_NAME_RECOGNIZER", "vit_ctc_deepseek"),
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
    mlflow_tracking_recognizer = MlflowTrackingRecognizer(
        mlflow_tracking=mlflow_tracking,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    mlflow_artifacts = MlflowArtifactsRecognizer(
        mlflow_tracking=mlflow_tracking,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    mlflow_model_registry = MlflowModelRegistryRecognizer(
        mlflow_model_registry=mlflow_registry,
        mlflow_artifact=mlflow_artifact,
        logger=logger,
        path_env_ml_traning=env_path,
    )
    recognizer = MLflowRecognizer(
        mlflow_tracking_recognizer=mlflow_tracking_recognizer,
        mlflow_artifacts=mlflow_artifacts,
        mlflow_model_registry=mlflow_model_registry,
        mlflow_tracking=mlflow_tracking,
        logger=logger,
    )
    recognizer.execute(
        git_commit=git_commit,
        checkpoint_name=["best_cer.pt", "last_checkpoint.pt"],
        goal=os.getenv("MLFLOW_RUN_GOAL_RECOGNIZER", "training"),
        tracking_level=os.getenv("MLFLOW_TRACKING_LEVEL_RECOGNIZER", "full"),
        pwd=pwd
    )
