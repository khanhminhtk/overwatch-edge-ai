from __future__ import annotations

import subprocess

from mlflow import MlflowClient

from src.modules.tracking.application.use_case.recognizer.artifacts import RecognizerMlflowArtifactStore
from src.modules.tracking.application.use_case.recognizer.registry import RecognizerMlflowModelRegistry
from src.modules.tracking.application.use_case.recognizer.tracking import RecognizerMlflowTracking
from src.modules.tracking.application.use_case.recognizer.workflow import RecognizerMlflowWorkflow
from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.tracking.mlflow import MlflowArtifact, MlflowRegistry, MlflowTrace, MlflowTracking
from src.platform.tracking.mlflow.config import MlflowConfig


def main() -> None:
    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()
    recognizer_config = ConfigLoader.load(
        RecognizerConfig,
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        section={
            "mlflow": None,
            "mlflow.recognizer": None,
        },
    )
    print(f"Loaded RecognizerConfig: {recognizer_config}")

    mlflow_config = ConfigLoader.load(
        MlflowConfig,
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        section="mlflow",
    )

    run = RecognizerRun(
        experiment_name="test_experiment",
        run_name="test_run",
        config=recognizer_config,
    )
    client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
    tracking = MlflowTracking(client=client, tracking_url=mlflow_config.tracking_uri)
    registry = MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri)
    artifact_store = MlflowArtifact(client=client, tracking_url=mlflow_config.tracking_uri)
    trace_port = MlflowTrace(client=client, tracking_url=mlflow_config.tracking_uri)

    workflow = RecognizerMlflowWorkflow(
        tracking=RecognizerMlflowTracking(
            run=run,
            hardware_info=recognizer_config.hardware,
            experiment_tracker=tracking,
            trace_port=trace_port,
            logger=Logger("RecognizerMlflowTracking"),
        ),
        artifacts=RecognizerMlflowArtifactStore(
            run=run,
            tracker=tracking,
            artifact_store=artifact_store,
            logger=Logger("RecognizerMlflowArtifactStore"),
        ),
        registry=RecognizerMlflowModelRegistry(
            run=run,
            registry=registry,
            tracker=tracking,
            logger=Logger("RecognizerMlflowModelRegistry"),
        ),
        logger=Logger("RecognizerMlflowWorkflow"),
    )
    model_version = workflow.execute(
        checkpoint_names=[
            recognizer_config.best_checkpoint_name,
            recognizer_config.last_checkpoint_name,
        ],
        pwd=pwd,
    )
    print(f"Registered recognizer model version: {model_version}")


if __name__ == "__main__":
    main()
