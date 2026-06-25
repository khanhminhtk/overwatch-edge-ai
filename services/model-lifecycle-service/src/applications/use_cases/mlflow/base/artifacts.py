from __future__ import annotations

from abc import ABC, abstractmethod

from src.applications.use_cases.mlflow.base.common import BaseMlflowUseCase
from src.infra.mlflow.mlflow_artifact import MlflowArtifact
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger


class BaseMlflowArtifactsUseCase(MlflowArtifact, BaseMlflowUseCase, ABC):
    def __init__(
        self,
        *,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str,
        checkpoint_dir_env_key: str,
        model_name_env_key: str,
        docker_image_env_key: str,
        dataset_zip_path_env_key: str,
        manifest_path_env_key: str,
        dataset_name_env_key: str,
        pipeline_name_env_key: str,
        pipeline_run_id_env_key: str,
    ):
        MlflowArtifact.__init__(self, mlflow_tracking.client)
        BaseMlflowUseCase.__init__(self, logger=logger, path_env_ml_traning=path_env_ml_traning)
        self._mlflow_tracking = mlflow_tracking
        self._checkpoint_dir = self._env(checkpoint_dir_env_key)
        self._deployment_stage = self._env("MLFLOW_DEPLOYMENT_STAGE", "training")
        self._docker_image = self._env(docker_image_env_key)
        self._dataset_zip_path = self._env(dataset_zip_path_env_key)
        self._manifest_path = self._env(manifest_path_env_key)
        self._gpu_host_name = self._env("GPU_HOST_NAME", "unknown_host")
        self._gpu_name = self._env("GPU_NAME", "unknown_gpu")
        self._gpu_memory_gb = self._env_float("GPU_MEMORY_GB", 0)
        self._cuda_version = self._env("CUDA_VERSION", "unknown_cuda_version")
        self._dataset_name = self._env(dataset_name_env_key, "unknown_dataset")
        self._driver_version = self._env("GPU_DRIVER_VERSION", "unknown_driver_version")
        self._framework = self._env("FRAMEWORK_NAME", "unknown_framework")
        self._pipeline_name = self._env(pipeline_name_env_key, "unknown_pipeline")
        self._pipeline_run_id = self._env(pipeline_run_id_env_key, "unknown_pipeline_run_id")
        self._model_name = self._env(model_name_env_key, "")

    def execute(
        self,
        checkpoint_name: list[str],
        run_id: str | None = None,
        pwd: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError(f"{self.__class__.__name__}.execute: run_id is required for MLflow artifact upload")
        resolved_pwd = pwd or ""
        self._upload_dataset_artifacts(run_id, resolved_pwd)
        self._upload_training_artifacts(run_id, resolved_pwd)
        self._upload_checkpoint_artifacts(run_id, checkpoint_name, resolved_pwd)
        self._log_common_params(run_id, resolved_pwd)

    def _log_common_params(self, run_id: str, pwd: str) -> None:
        params = {
            "deployment_stage": self._deployment_stage,
            "docker_image": self._docker_image,
            "dataset_name": self._dataset_name,
            "dataset_zip_size_gb": self._compute_dataset_zip_size_gb(pwd),
            "gpu_host_name": self._gpu_host_name,
            "gpu_name": self._gpu_name,
            "gpu_memory_gb": self._gpu_memory_gb,
            "cuda_version": self._cuda_version,
            "driver_version": self._driver_version,
            "framework": self._framework,
            "pipeline_name": self._pipeline_name,
            "pipeline_run_id": self._pipeline_run_id,
        }
        self._mlflow_tracking.log_params(run_id, params)

    def _compute_dataset_zip_size_gb(self, pwd: str) -> float:
        if not self._dataset_zip_path:
            return 0.0
        resolved_dataset_zip_path = self._resolve_path(pwd, self._dataset_zip_path)
        if not self._path_exists(resolved_dataset_zip_path):
            raise FileNotFoundError(f"Dataset zip file not found: {resolved_dataset_zip_path}")
        return self._path_size(resolved_dataset_zip_path) / (1024 ** 3)

    def _upload_dataset_artifacts(self, run_id: str, pwd: str) -> None:
        self.log_artifact(run_id, self._resolve_path(pwd, self._manifest_path), artifact_path="dataset")
        self.log_artifact(run_id, self._resolve_path(pwd, self._dataset_zip_path), artifact_path="dataset")

    @staticmethod
    def _path_exists(path: str) -> bool:
        import os

        return os.path.exists(path)

    @staticmethod
    def _path_size(path: str) -> int:
        import os

        return os.path.getsize(path)

    @abstractmethod
    def _upload_training_artifacts(self, run_id: str, pwd: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _upload_checkpoint_artifacts(self, run_id: str, checkpoint_name: list[str], pwd: str) -> None:
        raise NotImplementedError
