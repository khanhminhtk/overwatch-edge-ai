from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.ports.mlflow_artifact_port import MlflowArtifactPort
from src.applications.ports.mlflow_model_registry_port import MlflowModelRegistryPort
from src.applications.use_cases.mlflow.base.model_registry import BaseMlflowModelRegistryUseCase
from src.utils.logger import Logger


class MlflowModelRegistryDetection(BaseMlflowModelRegistryUseCase):
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
            model_name_env_key="MLFLOW_MODEL_NAME_DETECTION",
            best_checkpoint_env_key="DETECT_BEST_CHECKPOINT_NAME",
            checkpoint_dir_env_key="DETECT_SAVE_DIR",
            pipeline_name_env_key="PIPELINE_NAME_DETECTION",
            pipeline_run_id_env_key="PIPELINE_RUN_ID_DETECTION",
            dataset_name_env_key="DETECT_DATASET_NAME",
            model_alias_env_key="MLFLOW_MODEL_ALIAS_DETECTION",
            model_type_tag="detection",
        )

    def _load_model_metadata(self, pwd: str) -> dict[str, int | float | str | None]:
        import torch

        checkpoint_path = self._resolve_path(pwd, self._checkpoint_dir, self._best_checkpoint_name)
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
        train_metrics = checkpoint.get("train_metrics") or {}
        return {
            "epoch": checkpoint.get("epoch"),
            "best_fitness": checkpoint.get("best_fitness"),
            "version": checkpoint.get("version"),
            **train_metrics,
            "model_total_params": total_params,
        }

    @staticmethod
    def _serialize_model_metadata(
        checkpoint_metadata: dict[str, int | float | str | None],
    ) -> dict[str, str]:
        renamed_keys = {
            "epoch": "checkpoint_epoch",
            "version": "checkpoint_version",
        }
        serialized: dict[str, str] = {}
        for key, value in checkpoint_metadata.items():
            normalized_key = renamed_keys.get(key, MlflowModelRegistryDetection._normalize_metric_key(key))
            serialized[normalized_key] = str(value)
        return serialized

    @staticmethod
    def _normalize_metric_key(key: str) -> str:
        normalized = key.strip().replace("/", "_")
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_").lower()
