from __future__ import annotations

from abc import ABC, abstractmethod

from src.applications.ports.mlflow_artifact_port import MlflowArtifactPort
from src.applications.ports.mlflow_model_registry_port import MlflowModelRegistryPort
from src.applications.use_cases.mlflow.base.common import BaseMlflowUseCase
from src.utils.logger import Logger


class BaseMlflowModelRegistryUseCase(BaseMlflowUseCase, ABC):
    def __init__(
        self,
        *,
        mlflow_model_registry: MlflowModelRegistryPort,
        mlflow_artifact: MlflowArtifactPort,
        logger: Logger,
        path_env_ml_traning: str,
        model_name_env_key: str,
        best_checkpoint_env_key: str,
        checkpoint_dir_env_key: str,
        pipeline_name_env_key: str,
        pipeline_run_id_env_key: str,
        dataset_name_env_key: str,
        model_alias_env_key: str,
        model_type_tag: str,
    ):
        super().__init__(logger=logger, path_env_ml_traning=path_env_ml_traning)
        self._mlflow_model_registry = mlflow_model_registry
        self._mlflow_artifact = mlflow_artifact
        self._deployment_stage = self._env("MLFLOW_DEPLOYMENT_STAGE", "staging")
        self._model_name = self._env(model_name_env_key).strip()
        self._best_checkpoint_name = self._env(best_checkpoint_env_key).strip()
        self._alias_override = self._env(model_alias_env_key).strip()
        self._checkpoint_dir = self._env(checkpoint_dir_env_key).strip()
        self._gpu_host_name = self._env("GPU_HOST_NAME", "unknown_host")
        self._gpu_name = self._env("GPU_NAME", "unknown_gpu")
        self._dataset_name = self._env(dataset_name_env_key, "unknown_dataset")
        self._pipeline_name = self._env(pipeline_name_env_key, "unknown_pipeline")
        self._pipeline_run_id = self._env(pipeline_run_id_env_key, "unknown_pipeline_run_id")
        self._framework = self._env("FRAMEWORK_NAME", "unknown_framework")
        self._model_type_tag = model_type_tag

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
        model_uri = self._build_model_uri(normalized_run_id)
        checkpoint_metadata = self._load_model_metadata(pwd or "")
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

    def export_model_download_url(self, model_name: str, version: str) -> str:
        normalized_model_name = model_name.strip()
        normalized_version = version.strip()
        if not normalized_model_name:
            raise ValueError("model_name is required for MLflow model export")
        if not normalized_version:
            raise ValueError("version is required for MLflow model export")
        download_url = self._mlflow_model_registry.export_model_download_url(
            normalized_model_name,
            normalized_version,
        )
        self._logger.info(
            "[MLFLOW_MODEL_EXPORT_URL_RESOLVED]",
            f"model_name={normalized_model_name}",
            f"version={normalized_version}",
        )
        return download_url

    def _build_model_uri(self, run_id: str) -> str:
        return f"runs:/{run_id}/checkpoints/{self._best_checkpoint_name}"

    def _resolve_alias(self, deployment_stage: str) -> str:
        if self._alias_override:
            return self._alias_override
        normalized_stage = deployment_stage.strip().lower()
        if normalized_stage == "staging":
            return "candidate"
        if normalized_stage == "production":
            return "champion"
        return normalized_stage or "candidate"

    def _build_version_tags(
        self,
        run_id: str,
        git_commit: str,
        framework: str,
        checkpoint_metadata: dict[str, int | float | str | None],
    ) -> dict[str, str]:
        return {
            "git_commit": git_commit,
            "framework": framework,
            "model_type": self._model_type_tag,
            "deployment_stage": self._deployment_stage,
            "source_run_id": run_id,
            "host_name": self._gpu_host_name,
            "gpu_name": self._gpu_name,
            "dataset_name": self._dataset_name,
            "pipeline_name": self._pipeline_name,
            "pipeline_run_id": self._pipeline_run_id,
            **self._serialize_model_metadata(checkpoint_metadata),
        }

    @staticmethod
    def _serialize_model_metadata(
        checkpoint_metadata: dict[str, int | float | str | None],
    ) -> dict[str, str]:
        return {key: str(value) for key, value in checkpoint_metadata.items()}

    @abstractmethod
    def _load_model_metadata(self, pwd: str) -> dict[str, int | float | str | None]:
        raise NotImplementedError
