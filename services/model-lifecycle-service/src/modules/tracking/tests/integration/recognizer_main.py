from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from mlflow import MlflowClient

from src.modules.tracking.application.use_case.recognizer.artifacts import (
    RecognizerMlflowArtifactStore,
)
from src.modules.tracking.application.use_case.recognizer.registry import (
    RecognizerMlflowModelRegistry,
)
from src.modules.tracking.application.use_case.recognizer.tracking import (
    RecognizerMlflowTracking,
)
from src.modules.tracking.application.use_case.recognizer.workflow import (
    RecognizerMlflowWorkflow,
)
from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.tracking.mlflow import MlflowArtifact, MlflowRegistry, MlflowTrace, MlflowTracking
from src.platform.tracking.mlflow.config import MlflowConfig


REPO_ROOT = SERVICE_ROOT.parent.parent
CONFIG_YAML = REPO_ROOT / "services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"
CONFIG_ENV = REPO_ROOT / "services/model-lifecycle-service/config/.env"


def build_workflow(experiment_name: str) -> tuple[RecognizerMlflowWorkflow, RecognizerRun, MlflowClient]:
    detection_config = ConfigLoader.load(
        RecognizerConfig,
        yaml_files=[str(CONFIG_YAML)],
        env_files=[str(CONFIG_ENV)],
        section={
            "mlflow": None,
            "mlflow.recognizer": None,
        },
    )
    mlflow_config = ConfigLoader.load(
        MlflowConfig,
        yaml_files=[str(CONFIG_YAML)],
        env_files=[str(CONFIG_ENV)],
        section="mlflow",
    )
    run = RecognizerRun(
        experiment_name=experiment_name,
        run_name="tracking_recognizer_integration",
        config=detection_config,
    )
    client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
    logger = Logger("RecognizerTrackingIntegrationMain")
    workflow = RecognizerMlflowWorkflow(
        tracking=RecognizerMlflowTracking(
            run=run,
            hardware_info=detection_config.hardware,
            experiment_tracker=MlflowTracking(client=client, tracking_url=mlflow_config.tracking_uri),
            trace_port=MlflowTrace(client=client, tracking_url=mlflow_config.tracking_uri),
            logger=logger,
        ),
        artifacts=RecognizerMlflowArtifactStore(
            run=run,
            tracker=MlflowTracking(client=client, tracking_url=mlflow_config.tracking_uri),
            artifact_store=MlflowArtifact(client=client, tracking_url=mlflow_config.tracking_uri),
            logger=logger,
        ),
        registry=RecognizerMlflowModelRegistry(
            run=run,
            registry=MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri),
            tracker=MlflowTracking(client=client, tracking_url=mlflow_config.tracking_uri),
            logger=logger,
        ),
        logger=logger,
    )
    return workflow, run, client


def main() -> dict[str, Any]:
    experiment_name = f"tracking_recognizer_integration_{uuid.uuid4().hex[:8]}"
    workflow, run, client = build_workflow(experiment_name=experiment_name)
    model_version = workflow.execute(
        checkpoint_names=[run.config.best_checkpoint_name, run.config.last_checkpoint_name],
        pwd=str(REPO_ROOT),
    )
    return {
        "client": client,
        "experiment_name": experiment_name,
        "run_id": run.run_id,
        "model_version": model_version,
    }


if __name__ == "__main__":
    payload = main()
    print(
        json.dumps(
            {
                "experiment_name": payload["experiment_name"],
                "run_id": payload["run_id"],
                "model_version": payload["model_version"],
            }
        )
    )
