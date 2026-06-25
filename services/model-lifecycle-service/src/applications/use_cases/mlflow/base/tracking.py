from __future__ import annotations

from abc import ABC, abstractmethod

from src.applications.dtos.mlflow_production_run_metadata import MlflowRunTags
from src.applications.use_cases.mlflow.base.common import BaseMlflowUseCase
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger


class BaseMlflowTrackingUseCase(BaseMlflowUseCase, ABC):
    def __init__(
        self,
        *,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str,
        model_name_env_key: str,
        model_type_env_key: str,
        task_name_env_key: str,
        docker_image_env_key: str,
        dataset_zip_path_env_key: str,
        manifest_path_env_key: str,
    ):
        super().__init__(logger=logger, path_env_ml_traning=path_env_ml_traning)
        self._mlflow_tracking = mlflow_tracking
        self._deployment_stage = self._env("MLFLOW_DEPLOYMENT_STAGE", "training")
        self._docker_image = self._env(docker_image_env_key)
        self._dataset_zip_path = self._env(dataset_zip_path_env_key)
        self._manifest_path = self._env(manifest_path_env_key)
        self._gpu_host_name = self._env("GPU_HOST_NAME", "unknown_host")
        self._gpu_name = self._env("GPU_NAME", "unknown_gpu")
        self._gpu_memory_gb = self._env_float("GPU_MEMORY_GB", 0)
        self._framework = self._env("FRAMEWORK_NAME", "unknown_framework")
        self._model_name = self._env(model_name_env_key, "")
        self._task_name = self._env(task_name_env_key, "unknown_task")
        self._model_type = self._env(model_type_env_key, "unknown_model_type")

    def execute(
        self,
        git_commit: str,
        goal: str,
        tracking_level: str,
        run_id: str,
        pwd: str | None = None,
        source: str = "ml/training/scripts/train_recognizer.sh",
    ) -> None:
        self.export_training_metrics(run_id=run_id, pwd=pwd or "")
        self.log_tag(
            git_commit=git_commit,
            host_name=self._gpu_host_name,
            gpu_name=self._gpu_name,
            framework=self._framework,
            goal=goal,
            tracking_level=tracking_level,
            run_id=run_id,
            source=source,
        )

    @abstractmethod
    def export_training_metrics(self, run_id: str, pwd: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _metric_source_tag_value(self) -> str:
        raise NotImplementedError

    def log_tag(
        self,
        git_commit: str,
        host_name: str,
        gpu_name: str,
        framework: str,
        goal: str,
        tracking_level: str,
        run_id: str,
        source: str = "ml/training/scripts/train_recognizer.sh",
    ) -> None:
        tags = MlflowRunTags(
            source=source,
            goal=goal,
            tracking_level=tracking_level,
            model_type=self._model_type,
            task=self._task_name,
            registered_model_name=self._model_name,
            deployment_stage=self._deployment_stage,
            git_commit=git_commit,
            docker_image=self._docker_image,
            host_name=host_name,
            gpu_name=gpu_name,
            metadata_mocked="false",
            tensorboard_dir=self._metric_source_tag_value(),
            checkpoint_dir=self._checkpoint_dir_tag_value(),
            dataset_zip_path=self._dataset_zip_path,
            manifest_path=self._manifest_path,
            framework=framework,
        )
        self._mlflow_tracking.resume_run(run_id=run_id)
        self._mlflow_tracking.set_tags(run_id=run_id, tags=tags.__dict__)

    @abstractmethod
    def _checkpoint_dir_tag_value(self) -> str:
        raise NotImplementedError
