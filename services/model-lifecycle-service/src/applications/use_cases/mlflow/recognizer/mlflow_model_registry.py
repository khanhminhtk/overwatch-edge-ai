import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from mlflow.models import infer_signature

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.model_registry import BaseMlflowModelRegistryUseCase
from src.applications.ports.mlflow_artifact_port import MlflowArtifactPort
from src.applications.ports.mlflow_model_registry_port import MlflowModelRegistryPort
from src.applications.use_cases.mlflow.recognizer.pyfunc_model import RecognizerCheckpointPyfuncModel
from src.utils.logger import Logger

DEFAULT_ENV_PATH = (
    "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/"
    "ml/training/config/.env.example"
)


class MlflowModelRegistryRecognizer(BaseMlflowModelRegistryUseCase):
    def __init__(
        self,
        mlflow_model_registry: MlflowModelRegistryPort,
        mlflow_artifact: MlflowArtifactPort,
        logger: Logger,
        path_env_ml_traning: str,
    ):
        super().__init__(
            mlflow_model_registry=mlflow_model_registry,
            mlflow_artifact=mlflow_artifact,
            logger=logger,
            path_env_ml_traning=path_env_ml_traning,
            model_name_env_key="MLFLOW_MODEL_NAME_RECOGNIZER",
            best_checkpoint_env_key="RECOG_BEST_CHECKPOINT_NAME",
            checkpoint_dir_env_key="RECOG_SAVE_DIR",
            pipeline_name_env_key="PIPELINE_NAME_RECOGNIZER",
            pipeline_run_id_env_key="PIPELINE_RUN_ID_RECOGNIZER",
            dataset_name_env_key="RECOG_DATASET_NAME",
            model_alias_env_key="MLFLOW_MODEL_ALIAS_RECOGNIZER",
            model_type_tag="recognizer",
        )

    def _load_model_metadata(self, pwd: str) -> dict[str, int | float | str | None]:
        return self._load_checkpoint_metadata(pwd)

    def execute(
        self,
        run_id: str,
        git_commit: str,
        framework: str | None = None,
        pwd: str | None = None,
    ) -> str:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id is required for MLflow model registration")
        if not self._model_name:
            raise ValueError("registered model name is required")
        if not self._best_checkpoint_name:
            raise ValueError("best checkpoint name is required")

        resolved_framework = framework or self._framework
        resolved_pwd = pwd or ""
        checkpoint_metadata = self._load_checkpoint_metadata(resolved_pwd)
        model_uri = self._log_pyfunc_model(
            run_id=normalized_run_id,
            framework=resolved_framework,
            checkpoint_metadata=checkpoint_metadata,
            pwd=resolved_pwd,
        )
        model_version = self._mlflow_model_registry.register_model(
            model_uri=model_uri,
            registered_model_name=self._model_name,
        )
        self._mlflow_model_registry.set_model_version_tags(
            registered_model_name=self._model_name,
            model_version=model_version,
            tags=self._build_version_tags(
                run_id=normalized_run_id,
                git_commit=git_commit,
                framework=resolved_framework,
                checkpoint_metadata=checkpoint_metadata,
            ),
        )

        alias = self._resolve_alias(self._deployment_stage)
        self._mlflow_model_registry.set_model_alias(
            registered_model_name=self._model_name,
            alias=alias,
            model_version=model_version,
        )
        self._mlflow_model_registry.transition_model_stage(
            registered_model_name=self._model_name,
            model_version=model_version,
            stage=self._deployment_stage,
        )
        self._mlflow_artifact.log_json(
            normalized_run_id,
            {
                "registered_model_name": self._model_name,
                "model_version": model_version,
                "model_uri": model_uri,
                "source_checkpoint_uri": f"runs:/{normalized_run_id}/checkpoints/{self._best_checkpoint_name}",
                "deployment_stage": self._deployment_stage,
                "alias": alias,
                "git_commit": git_commit,
                "framework": resolved_framework,
                "host_name": self._gpu_host_name,
                "gpu_name": self._gpu_name,
                "dataset_name": self._dataset_name,
                "pipeline_name": self._pipeline_name,
                "pipeline_run_id": self._pipeline_run_id,
                "checkpoint": checkpoint_metadata,
            },
            "production/registry_summary.json",
        )
        self._logger.info(
            "[MLFLOW_MODEL_REGISTERED]",
            f"run_id={normalized_run_id}",
            f"registered_model_name={self._model_name}",
            f"model_version={model_version}",
            f"deployment_stage={self._deployment_stage}",
            f"alias={alias}",
        )
        return model_version

    def _log_pyfunc_model(
        self,
        run_id: str,
        framework: str,
        checkpoint_metadata: dict[str, int | float | str | None],
        pwd: str,
    ) -> str:
        checkpoint_path = self._resolve_path(pwd, self._checkpoint_dir, self._best_checkpoint_name)
        pyfunc_model = RecognizerCheckpointPyfuncModel(
            checkpoint_path=checkpoint_path,
            metadata=checkpoint_metadata,
        )
        input_example = pd.DataFrame(
            [
                {
                    "image_path": "sample.png",
                    "request_id": "example-request",
                }
            ]
        )
        output_example = pyfunc_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)
        return self._mlflow_model_registry.log_pyfunc_model(
            run_id=run_id,
            model_name="recognizer_pyfunc_model",
            python_model=pyfunc_model,
            input_example=input_example,
            signature=signature,
            tags={
                "framework": framework,
                "model_type": "recognizer",
            },
            params={
                "checkpoint_name": self._best_checkpoint_name,
                "framework": framework,
                "dataset_name": self._dataset_name,
            },
        )

    def _load_checkpoint_metadata(self, pwd: str) -> dict[str, int | float | str | None]:
        import torch

        checkpoint_path = self._resolve_path(pwd, self._checkpoint_dir, self._best_checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder_vocab = checkpoint.get("encoder_vocab") or []
        encoder_state_dict = checkpoint.get("encoder_state_dict") or {}
        return {
            "epoch": checkpoint.get("epoch"),
            "version": checkpoint.get("version"),
            "epoch_idx": checkpoint.get("epoch_idx"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "best_val_cer": checkpoint.get("best_val_cer"),
            "encoder_num_classes": checkpoint.get("encoder_num_classes"),
            "encoder_vocab_size": len(encoder_vocab),
            "model_total_params": sum(param.numel() for param in encoder_state_dict.values()),
        }

    @staticmethod
    def _serialize_model_metadata(
        checkpoint_metadata: dict[str, int | float | str | None],
    ) -> dict[str, str]:
        renamed_keys = {
            "epoch": "checkpoint_epoch",
            "version": "checkpoint_version",
        }
        return {
            renamed_keys.get(key, key): str(value)
            for key, value in checkpoint_metadata.items()
            if key != "epoch_idx"
        }


def resolve_run_id(run_id: str) -> str:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("MLFLOW_RUN_ID_RECOGNIZER_TEST is required to run registry test flow")
    return normalized_run_id


# if __name__ == "__main__":
#     from src.infra.mlflow.mlflow_artifact import MlflowArtifact
#     from src.infra.mlflow.mlflow_registry import MlflowRegistry
#     from src.infra.mlflow.mlflow_tracking import MlflowTracking

#     tracking_uri = "http://localhost:5000"
#     env_path = DEFAULT_ENV_PATH
#     load_dotenv(dotenv_path=env_path)
#     git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
#     framework = os.getenv("FRAMEWORK_NAME", "pytorch")

#     logger = Logger()
#     mlflow_registry = MlflowRegistry(tracking_uri=tracking_uri)
#     mlflow_artifact = MlflowArtifact(mlflow_registry.client)
#     mlflow_tracking = MlflowTracking(tracking_uri=tracking_uri)
#     run_id = mlflow_tracking.start_run(
#         experiment_name="recognizer_training-test",
#         run_name=f"recognizer_training_{git_commit}_{os.getenv('GPU_HOST_NAME', 'unknown_host')}_{os.getenv('GPU_NAME', 'unknown_gpu')}"
#     )
#     recognizer = MlflowModelRegistryRecognizer(
#         mlflow_model_registry=mlflow_registry,
#         mlflow_artifact=mlflow_artifact,
#         logger=logger,
#         path_env_ml_traning=env_path,
#     )
#     model_version = recognizer.execute(
#         run_id=run_id,
#         git_commit=git_commit
#     )
#     print(f"Registered model version: {model_version}")
