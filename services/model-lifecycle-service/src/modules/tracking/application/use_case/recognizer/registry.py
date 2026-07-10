from __future__ import annotations

from typing import Any

from pathlib import Path

import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.application.use_case.tracking_common import log_exceptions
from src.platform.logger import Logger
from src.platform.tracking.mlflow import ExperimentTracker, ModelRegistry


class RecognizerCheckpointPyfuncModel(PythonModel):
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
                "text": [""] * num_rows,
                "confidence": [0.0] * num_rows,
                "checkpoint_path": [self._checkpoint_path] * num_rows,
            }
        )


class RecognizerMlflowModelRegistry:
    def __init__(
        self,
        run: RecognizerRun,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        logger: Logger,
    ) -> None:
        self.run = run
        self._registry = registry
        self._tracker = tracker
        self._logger = logger

    @log_exceptions("[RECOGNIZER_REGISTRY_EXECUTE_ERROR]")
    def execute(self, run_id: str, pwd: str) -> str:
        return self.log_model_version_info(run_id=run_id, pwd=pwd)

    @log_exceptions("[RECOGNIZER_REGISTRY_RESOLVE_PATH_ERROR]")
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
                    "image_path": "sample.png",
                    "request_id": "example-request",
                }
            ]
        )

    @log_exceptions("[RECOGNIZER_REGISTRY_BUILD_RUN_PARAMS_ERROR]")
    def _build_run_params(self, checkpoint_loaded: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch_best": checkpoint_loaded.get("epoch_best"),
            "epoch_last": checkpoint_loaded.get("epoch_last"),
        }

    @log_exceptions("[RECOGNIZER_REGISTRY_BUILD_RUN_METRICS_ERROR]")
    def _build_run_metrics(self, checkpoint_loaded: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key in ("best_val_loss", "best_val_cer"):
            value = checkpoint_loaded.get(key)
            if isinstance(value, int | float):
                metrics[key] = float(value)
        return metrics

    @log_exceptions("[RECOGNIZER_REGISTRY_EXTRACT_CER_ERROR]")
    def _extract_best_val_cer(self, checkpoint_loaded: dict[str, Any]) -> float | None:
        value = checkpoint_loaded.get("best_val_cer")
        if isinstance(value, int | float):
            return float(value)
        return None

    @log_exceptions("[RECOGNIZER_REGISTRY_PROMOTE_ERROR]")
    def _promote_if_better(
        self,
        model_name: str,
        model_version: str,
        best_val_cer: float | None,
    ) -> None:
        if best_val_cer is None:
            return

        champion = self._registry.get_model_version_by_alias(model_name, "champion")
        if champion is None:
            self._registry.set_model_alias(model_name, "champion", model_version)
            self._registry.transition_model_stage(model_name, model_version, "Production")
            return

        champion_cer_raw = getattr(champion, "tags", {}).get("best_val_cer")
        try:
            champion_cer = float(champion_cer_raw)
        except (TypeError, ValueError):
            champion_cer = None

        if champion_cer is None or best_val_cer < champion_cer:
            self._registry.set_model_alias(model_name, "champion", model_version)
            self._registry.transition_model_stage(model_name, model_version, "Production")

    @log_exceptions("[RECOGNIZER_REGISTRY_LOG_MODEL_INFO_ERROR]")
    def log_model_version_info(self, run_id: str, pwd: str) -> str:
        checkpoint_loaded = self._load_checkpoint_bundle(pwd)
        checkpoint_path = str(
            self._resolve_path(
                pwd,
                self.run.config.checkpoint_dir,
                self.run.config.best_checkpoint_name,
            )
        )
        input_example = self._build_input_example()
        python_model = RecognizerCheckpointPyfuncModel(checkpoint_path=checkpoint_path)
        output_example = python_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)
        run_params = self._build_run_params(checkpoint_loaded)
        run_metrics = self._build_run_metrics(checkpoint_loaded)
        best_val_cer = self._extract_best_val_cer(checkpoint_loaded)
        self._tracker.log_params(run_id=run_id, params=run_params)
        self._tracker.log_metrics(run_id=run_id, metrics=run_metrics)
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
        if best_val_cer is not None:
            self._registry.set_model_version_tags(
                registered_model_name=self.run.config.model_name,
                model_version=model_version,
                tags={"best_val_cer": str(best_val_cer)},
            )
        self._promote_if_better(
            model_name=self.run.config.model_name,
            model_version=model_version,
            best_val_cer=best_val_cer,
        )
        return model_version

    @log_exceptions("[RECOGNIZER_REGISTRY_LOAD_CHECKPOINT_BUNDLE_ERROR]")
    def _load_checkpoint_bundle(self, pwd: str) -> dict[str, Any]:
        import torch

        cfg = self.run.config
        best_checkpoint_path = self._resolve_path(pwd, cfg.checkpoint_dir, cfg.best_checkpoint_name)
        last_checkpoint_path = self._resolve_path(pwd, cfg.checkpoint_dir, cfg.last_checkpoint_name)
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {best_checkpoint_path}")
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {last_checkpoint_path}")

        best_checkpoint = torch.load(str(best_checkpoint_path), map_location="cpu", weights_only=False)
        last_checkpoint = torch.load(str(last_checkpoint_path), map_location="cpu", weights_only=False)
        encoder_vocab = best_checkpoint.get("encoder_vocab") or []
        encoder_state_dict = best_checkpoint.get("encoder_state_dict") or {}
        return {
            "epoch_best": best_checkpoint.get("epoch"),
            "epoch_last": last_checkpoint.get("epoch"),
            "version": best_checkpoint.get("version"),
            "best_val_loss": best_checkpoint.get("best_val_loss"),
            "best_val_cer": best_checkpoint.get("best_val_cer"),
            "encoder_num_classes": best_checkpoint.get("encoder_num_classes"),
            "encoder_vocab_size": len(encoder_vocab),
            "model_total_params": sum(param.numel() for param in encoder_state_dict.values()),
        }
