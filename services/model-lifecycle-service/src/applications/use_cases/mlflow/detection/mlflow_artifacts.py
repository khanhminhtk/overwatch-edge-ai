from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.artifacts import BaseMlflowArtifactsUseCase
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger


class MlflowArtifactsDetection(BaseMlflowArtifactsUseCase):
    def __init__(
        self,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str,
    ):
        super().__init__(
            mlflow_tracking=mlflow_tracking,
            logger=logger,
            path_env_ml_traning=path_env_ml_traning,
            checkpoint_dir_env_key="DETECT_SAVE_DIR",
            model_name_env_key="MLFLOW_MODEL_NAME_DETECTION",
            docker_image_env_key="DETECT_DOCKER_IMAGE",
            dataset_zip_path_env_key="DETECT_DATASET_ZIP_PATH",
            manifest_path_env_key="DETECT_MANIFEST_PATH",
            dataset_name_env_key="DETECT_DATASET_NAME",
            pipeline_name_env_key="PIPELINE_NAME_DETECTION",
            pipeline_run_id_env_key="PIPELINE_RUN_ID_DETECTION",
        )
        self._results_csv_path = self._env("DETECT_RESULTS_CSV_PATH")
        self._best_checkpoint_name = self._env("DETECT_BEST_CHECKPOINT_NAME", "best.pt")
        self._last_checkpoint_name = self._env("DETECT_LAST_CHECKPOINT_NAME", "last.pt")

    def _upload_training_artifacts(self, run_id: str, pwd: str) -> None:
        if self._results_csv_path:
            self.log_artifact(
                run_id,
                self._resolve_path(pwd, self._results_csv_path),
                artifact_path="training",
            )

    def _upload_checkpoint_artifacts(self, run_id: str, checkpoint_name: list[str], pwd: str) -> None:
        for name in checkpoint_name:
            self.log_artifact(
                run_id,
                self._resolve_path(pwd, self._checkpoint_dir, name),
                artifact_path="checkpoints",
            )
            checkpoint = self._load_checkpoint_metadata(self._resolve_path(pwd, self._checkpoint_dir), name)
            if name == self._last_checkpoint_name:
                self._mlflow_tracking.log_params(run_id, self._build_last_checkpoint_params(checkpoint))
            if name == self._best_checkpoint_name:
                self._mlflow_tracking.log_params(
                    run_id,
                    self._build_best_checkpoint_params(checkpoint),
                )

    def _load_checkpoint_metadata(self, checkpoint_dir: str, file_checkpoint: str) -> dict[str, int | float | str | None]:
        import torch

        checkpoint_path = os.path.join(checkpoint_dir, file_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model = checkpoint.get("model")
        total_params = 0
        if model is not None and hasattr(model, "parameters"):
            total_params = sum(param.numel() for param in model.parameters())
        return {
            "epoch": checkpoint.get("epoch"),
            "best_fitness": checkpoint.get("best_fitness"),
            "version": checkpoint.get("version"),
            "model_total_params": total_params,
            "train_metrics": checkpoint.get("train_metrics") or {},
            "train_results": checkpoint.get("train_results") or {},
        }

    @staticmethod
    def _normalize_metric_key(key: str) -> str:
        normalized = key.strip().replace("/", "_")
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_").lower()

    def _build_last_checkpoint_params(
        self,
        checkpoint: dict[str, object],
    ) -> dict[str, int | float]:
        params: dict[str, int | float] = {}
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, int | float):
            params["epoch"] = int(epoch)

        train_metrics = checkpoint.get("train_metrics")
        if isinstance(train_metrics, dict):
            for key, value in train_metrics.items():
                if isinstance(value, int | float):
                    params[f"train_metrics.{self._normalize_metric_key(str(key))}"] = float(value)

        train_results = checkpoint.get("train_results")
        if isinstance(train_results, dict):
            for key, value in train_results.items():
                if key == "epoch":
                    continue
                if isinstance(value, list) and value:
                    last_value = value[-1]
                    if isinstance(last_value, int | float):
                        params[f"train_results_last.{self._normalize_metric_key(str(key))}"] = float(last_value)
        return params

    def _build_best_checkpoint_params(
        self,
        checkpoint: dict[str, object],
    ) -> dict[str, int | float]:
        params: dict[str, int | float] = {}
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, int | float):
            params["epoch_best_map50"] = int(epoch)

        train_metrics = checkpoint.get("train_metrics")
        if isinstance(train_metrics, dict):
            for key, value in train_metrics.items():
                if isinstance(value, int | float):
                    params[f"best_metrics.{self._normalize_metric_key(str(key))}"] = float(value)
        return params
