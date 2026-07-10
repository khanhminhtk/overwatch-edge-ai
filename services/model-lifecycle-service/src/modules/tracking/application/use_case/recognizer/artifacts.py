from __future__ import annotations

import os
from pathlib import Path

from src.modules.tracking.domain.value_objects import RecognizerConfig, HardwareInfo
from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.application.use_case.tracking_common import log_exceptions
from src.platform.tracking.mlflow import ExperimentTracker, ArtifactStore
from src.platform.logger import Logger


class RecognizerMlflowArtifactStore:
    def __init__(
        self,
        run: RecognizerRun,
        tracker: ExperimentTracker,
        artifact_store: ArtifactStore,
        logger: Logger,
    ) -> None:
        self.run = run
        self._experiment_tracker = tracker
        self._artifact_store = artifact_store
        self._logger = logger

    @log_exceptions("[RECOGNIZER_ARTIFACTS_EXECUTE_ERROR]")
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

    @log_exceptions("[RECOGNIZER_ARTIFACTS_RESOLVE_PATH_ERROR]")
    def _resolve_path(self, pwd: str, *parts: str) -> Path:
        resolved = Path(parts[0])
        if not resolved.is_absolute():
            resolved = Path(pwd) / resolved
        for part in parts[1:]:
            resolved = resolved / part
        return resolved

    @log_exceptions("[RECOGNIZER_ARTIFACTS_UPLOAD_CHECKPOINTS_ERROR]")
    def _upload_checkpoint_artifacts(self, run_id: str, checkpoint_name: list[str], pwd: str) -> None:
        cfg = self.run.config
        for name in checkpoint_name:
            self._artifact_store.log_artifact(
                run_id,
                str(self._resolve_path(pwd, cfg.checkpoint_dir, name)),
                artifact_path="checkpoints",
            )
            checkpoint = self._load_checkpoint(str(self._resolve_path(pwd, cfg.checkpoint_dir)), name)
            if name == cfg.last_checkpoint_name:
                self._experiment_tracker.log_params(run_id, checkpoint)
            if name == cfg.best_checkpoint_name:
                self._experiment_tracker.log_params(
                    run_id,
                    {
                        "epoch_best_val_cer": checkpoint.get("epoch"),
                    },
                )

    @log_exceptions("[RECOGNIZER_ARTIFACTS_UPLOAD_TRAINING_ERROR]")
    def _upload_training_artifacts(self, run_id: str, pwd: str) -> None:
        cfg = self.run.config
        training_artifact_dir = self._resolve_path(pwd, cfg.tensorboard_dir)
        for name in os.listdir(str(training_artifact_dir)):
            self._artifact_store.log_artifact(
                run_id,
                str(self._resolve_path(pwd, cfg.tensorboard_dir, name)),
                artifact_path="tensorboard",
            )

    @log_exceptions("[RECOGNIZER_ARTIFACTS_LOAD_CHECKPOINT_ERROR]")
    def _load_checkpoint(self, checkpoint_dir: str, file_checkpoint: str) -> dict[str, int | float | str | None]:
        import torch

        checkpoint_path = os.path.join(checkpoint_dir, file_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return {
            "epoch": checkpoint.get("epoch"),
            "version": checkpoint.get("version"),
            "epoch_idx": checkpoint.get("epoch_idx"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "best_val_cer": checkpoint.get("best_val_cer"),
            "encoder_vocab": checkpoint.get("encoder_vocab"),
            "encoder_num_classes": checkpoint.get("encoder_num_classes"),
            "num_encoder_vocab": len(checkpoint.get("encoder_vocab")) if checkpoint.get("encoder_vocab") else 0,
            "num_parameters": sum(p.numel() for p in checkpoint.get("encoder_state_dict").values()) if checkpoint.get("encoder_state_dict") else 0,
        }

    @log_exceptions("[RECOGNIZER_ARTIFACTS_LOG_COMMON_PARAMS_ERROR]")
    def _log_common_params(self, run_id: str, pwd: str) -> None:
        cfg = self.run.config
        hw = cfg.hardware
        params = {
            "deployment_stage": cfg.deployment_stage,
            "docker_image": cfg.docker_image,
            "dataset_name": cfg.dataset_name,
            "dataset_zip_size_gb": self._compute_dataset_zip_size_gb(pwd),
            "gpu_host_name": hw.gpu_host_name,
            "gpu_name": hw.gpu_name,
            "gpu_memory_gb": hw.gpu_memory_gb,
            "cuda_version": hw.cuda_version,
            "driver_version": hw.driver_version,
            "framework": cfg.framework,
            "pipeline_name": cfg.pipeline_name,
            "pipeline_run_id": cfg.pipeline_run_id,
        }
        self._experiment_tracker.log_params(run_id, params)

    @log_exceptions("[RECOGNIZER_ARTIFACTS_COMPUTE_DATASET_SIZE_ERROR]")
    def _compute_dataset_zip_size_gb(self, pwd: str) -> float:
        dataset_zip_path = self.run.config.dataset_zip_path
        if not dataset_zip_path:
            return 0.0
        resolved_dataset_zip_path = self._resolve_path(pwd, dataset_zip_path)
        if not os.path.exists(str(resolved_dataset_zip_path)):
            raise FileNotFoundError(f"Dataset zip file not found: {resolved_dataset_zip_path}")
        return os.path.getsize(str(resolved_dataset_zip_path)) / (1024 ** 3)

    @log_exceptions("[RECOGNIZER_ARTIFACTS_UPLOAD_DATASET_ERROR]")
    def _upload_dataset_artifacts(self, run_id: str, pwd: str) -> None:
        cfg = self.run.config
        if cfg.manifest_path:
            self._artifact_store.log_artifact(
                run_id,
                str(self._resolve_path(pwd, cfg.manifest_path)),
                artifact_path="dataset",
            )
        if cfg.dataset_zip_path:
            self._artifact_store.log_artifact(
                run_id,
                str(self._resolve_path(pwd, cfg.dataset_zip_path)),
                artifact_path="dataset",
            )

    @log_exceptions("[RECOGNIZER_ARTIFACTS_TRACK_ERROR]")
    def track_artifacts(self, recognizer_id: str, event: str, data: dict, pwd: str) -> None:
        self._logger.info(
            f"Tracking recognizer event: recognizer_id={recognizer_id}, event={event}, data={data}"
        )
        run_id = self._experiment_tracker.start_run(
            experiment_name=self.run.experiment_name,
            run_name=self.run.run_name,
        )
        self._experiment_tracker.resume_run(run_id)
        self.execute(
            checkpoint_name=[self.run.config.best_checkpoint_name, self.run.config.last_checkpoint_name],
            run_id=run_id,
            pwd=pwd,
        )
        self._experiment_tracker.end_run(run_id)


# if __name__ == "__main__":
#     import subprocess

#     from mlflow import MlflowClient

#     from src.platform.config import ConfigLoader
#     from src.platform.tracking.mlflow import ExperimentTracker, TracePort
#     from src.platform.logger import Logger
#     from src.platform.tracking.mlflow.config import MlflowConfig
#     from src.platform.tracking.mlflow import MlflowTracking, MlflowArtifact
#     from src.modules.tracking.domain.value_objects import RecognizerConfig, HardwareInfo
#     from src.modules.tracking.domain.entity_objects import RecognizerRun


#     pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()

#     recognizer_config = ConfigLoader.load(
#         RecognizerConfig,
#         yaml_files = [f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files = [f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section={
#             "mlflow": None,
#             "mlflow.recognizer": None,
#         }
#     )

#     print(f"Loaded RecognizerConfig: {recognizer_config}")

#     mlflow_config = ConfigLoader.load(
#         MlflowConfig,
#         yaml_files = [f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files = [f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section="mlflow"
#     )

#     run = RecognizerRun(
#         experiment_name="test_experiment",
#         run_name="test_run",
#         config=recognizer_config,
#     )

#     client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)

#     mlflow_tracking = MlflowTracking(
#         client=client,
#         tracking_url=mlflow_config.tracking_uri,
#     )

#     mlflow_artifact = MlflowArtifact(
#         client=client,
#         tracking_url=mlflow_config.tracking_uri,
#     )

#     logger = Logger("RecognizerMlflowTracking")

#     recognizer_mlflow_artifact_store = RecognizerMlflowArtifactStore(
#         run=run,
#         tracker=mlflow_tracking,
#         artifact_store=mlflow_artifact,
#         logger=logger,
#     )

#     recognizer_mlflow_artifact_store.track_artifacts(
#         recognizer_id="test_recognizer_id",
#         event="test_event",
#         data={"key": "value"},
#         pwd=pwd
#     )
