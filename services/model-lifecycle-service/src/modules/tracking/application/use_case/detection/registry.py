from __future__ import annotations
from typing import Any

from pathlib import Path

import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

from src.modules.tracking.domain.value_objects import DetectionConfig, HardwareInfo
from src.modules.tracking.domain.entity_objects import DetectionRun
from src.modules.tracking.application.use_case.tracking_common import log_exceptions
from src.platform.tracking.mlflow import ExperimentTracker, ModelRegistry
from src.platform.logger import Logger


class DetectionCheckpointPyfuncModel(PythonModel):
    def __init__(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        input_frame = pd.DataFrame(model_input)
        num_rows = len(input_frame.index)
        return pd.DataFrame(
            {
                "detections": [[] for _ in range(num_rows)],
                "checkpoint_path": [self._checkpoint_path] * num_rows,
            }
        )

class DetectionMlflowModelRegistry:
    def __init__(
        self,
        run: DetectionRun,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        logger: Logger,
    ) -> None:
        self.run = run
        self._registry = registry
        self._tracker = tracker
        self._logger = logger

    @log_exceptions("[DETECTION_REGISTRY_EXECUTE_ERROR]")
    def execute(self, run_id: str, pwd: str) -> str:
        return self.log_model_version_info(run_id=run_id, pwd=pwd)

    @log_exceptions("[DETECTION_REGISTRY_RESOLVE_PATH_ERROR]")
    def _resolve_path(self, pwd: str, *parts: str) -> Path:
        resolved = Path(parts[0])
        if not resolved.is_absolute():
            resolved = Path(pwd) / resolved
        for part in parts[1:]:
            resolved = resolved / part
        return resolved

    @staticmethod
    def _build_input_example() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "image_path": "sample.jpg",
                    "request_id": "example-request",
                }
            ]
        )

    @staticmethod
    def _normalize_metric_name(name: str) -> str:
        return (
            name.strip()
            .replace("metrics/", "")
            .replace("val/", "val_")
            .replace("(", "_")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_")
            .lower()
        )

    @log_exceptions("[DETECTION_REGISTRY_BUILD_RUN_PARAMS_ERROR]")
    def _build_run_params(self, checkpoint_loaded: dict[str, Any]) -> dict[str, Any]:
        return {
            "hardware": self.run.config.hardware.gpu_name,
            "epoch": checkpoint_loaded.get("epoch"),
        }

    @log_exceptions("[DETECTION_REGISTRY_BUILD_RUN_METRICS_ERROR]")
    def _build_run_metrics(self, checkpoint_loaded: dict[str, Any]) -> dict[str, float]:
        train_metrics = checkpoint_loaded.get("train_metrics") or {}
        metrics: dict[str, float] = {}
        for key, value in train_metrics.items():
            if isinstance(value, int | float):
                metrics[self._normalize_metric_name(str(key))] = float(value)
        return metrics

    @log_exceptions("[DETECTION_REGISTRY_EXTRACT_MAP50_ERROR]")
    def _extract_map50(self, checkpoint_loaded: dict[str, Any]) -> float | None:
        train_metrics = checkpoint_loaded.get("train_metrics") or {}
        map50 = train_metrics.get("metrics/mAP50(B)")
        if isinstance(map50, int | float):
            return float(map50)
        return None

    @log_exceptions("[DETECTION_REGISTRY_PROMOTE_ERROR]")
    def _promote_if_better(
        self,
        model_name: str,
        model_version: str,
        map50: float | None,
    ) -> None:
        if map50 is None:
            self._logger.info(
                "[MLFLOW_MODEL_PROMOTION_SKIPPED]",
                f"registered_model_name={model_name}",
                f"model_version={model_version}",
                "reason=missing_map50",
            )
            return

        champion = self._registry.get_model_version_by_alias(model_name, "champion")
        if champion is None:
            self._registry.set_model_alias(model_name, "champion", model_version)
            self._registry.transition_model_stage(model_name, model_version, "Production")
            self._logger.info(
                "[MLFLOW_MODEL_PROMOTED]",
                f"registered_model_name={model_name}",
                f"model_version={model_version}",
                f"map50={map50}",
                "reason=no_existing_champion",
            )
            return

        champion_map50_raw = getattr(champion, "tags", {}).get("map50")
        try:
            champion_map50 = float(champion_map50_raw)
        except (TypeError, ValueError):
            champion_map50 = None

        if champion_map50 is None or map50 > champion_map50:
            self._registry.set_model_alias(model_name, "champion", model_version)
            self._registry.transition_model_stage(model_name, model_version, "Production")
            self._logger.info(
                "[MLFLOW_MODEL_PROMOTED]",
                f"registered_model_name={model_name}",
                f"model_version={model_version}",
                f"map50={map50}",
                f"previous_champion_map50={champion_map50_raw}",
                "reason=better_map50",
            )
            return

        reason = "tie_map50" if map50 == champion_map50 else "lower_map50"
        self._logger.info(
            "[MLFLOW_MODEL_PROMOTION_SKIPPED]",
            f"registered_model_name={model_name}",
            f"model_version={model_version}",
            f"map50={map50}",
            f"champion_map50={champion_map50}",
            f"reason={reason}",
        )

    @log_exceptions("[DETECTION_REGISTRY_LOG_MODEL_INFO_ERROR]")
    def log_model_version_info(self, run_id: str, pwd: str) -> str:
        checkpoint_path = str(
            self._resolve_path(
                pwd,
                self.run.config.checkpoint_dir,
                self.run.config.best_checkpoint_name,
            )
        )
        checkpoint_loaded = self._load_checkpoint(
            checkpoint_dir=str(self._resolve_path(pwd, self.run.config.checkpoint_dir)),
            checkpoint_name=self.run.config.best_checkpoint_name,
        )
        input_example = self._build_input_example()
        python_model = DetectionCheckpointPyfuncModel(checkpoint_path=checkpoint_path)
        output_example = python_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)
        run_params = self._build_run_params(checkpoint_loaded)
        run_metrics = self._build_run_metrics(checkpoint_loaded)
        map50 = self._extract_map50(checkpoint_loaded)
        self._tracker.log_params(run_id=run_id, params=run_params)
        self._tracker.log_metrics(run_id=run_id, metrics=run_metrics)
        self._logger.info(f"Logging model version info for run_id={run_id}, checkpoint_path={checkpoint_path}, checkpoint_loaded_keys={list(checkpoint_loaded.keys()) if isinstance(checkpoint_loaded, dict) else checkpoint_loaded}")
        model_uri = self._registry.log_pyfunc_model(
            run_id=run_id,
            model_name=self.run.config.model_name,
            python_model=python_model,
            input_example=input_example,
            signature=signature,
            tags={"run_id": run_id},
            params=run_params,
        )
        model_version = self._registry.register_model(
            model_uri=model_uri,
            registered_model_name=self.run.config.model_name,
            run_id=run_id,
        )
        if map50 is not None:
            self._registry.set_model_version_tags(
                registered_model_name=self.run.config.model_name,
                model_version=model_version,
                tags={"map50": str(map50)},
            )
        self._promote_if_better(
            model_name=self.run.config.model_name,
            model_version=model_version,
            map50=map50,
        )
        return model_version

    @log_exceptions("[DETECTION_REGISTRY_LOAD_CHECKPOINT_ERROR]")
    def _load_checkpoint(self, checkpoint_dir: str, checkpoint_name: str) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint
