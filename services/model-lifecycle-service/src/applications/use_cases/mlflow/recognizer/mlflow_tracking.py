import os
from dotenv import load_dotenv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.applications.dtos.mlflow_production_run_metadata import MlflowRunTags
from src.utils.logger import Logger

SERVICE_ROOT = Path(__file__).resolve().parents[5]

class MlflowTrackingRecognizer:
    def __init__(
        self,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str
    ):
        self._mlflow_tracking = mlflow_tracking
        self._logger = logger
        load_dotenv(dotenv_path=path_env_ml_traning)
        self._tb = os.getenv("RECOG_TENSORBOARD_DIR")

    def excute(self, git_commit: str, host_name: str, gpu_name: str, framework: str, goal: str, tracking_level: str) -> None:
        run_id = self._mlflow_tracking.start_run(
            experiment_name="recognizer_training",
            run_name=f"recognizer_training_{git_commit}_{host_name}_{gpu_name}"
        )
        self.export_tensorboard_to_mlflow(run_id)
        self.log_tag(
            git_commit=git_commit,
            host_name=host_name,
            gpu_name=gpu_name,
            framework=framework,
            goal=goal,
            tracking_level=tracking_level,
            run_id=run_id
        )

    def export_tensorboard_to_mlflow(self, run_id: str) -> None:
        if not self._tb:
            raise ValueError("RECOG_TENSORBOARD_DIR is not configured")

        self._logger.info("[MLFLOW_TENSORBOARD_EXPORT_STARTED]", f"run_id={run_id}", f"tb_dir={self._tb}")

        event_acc = EventAccumulator(str(self._tb))
        event_acc.Reload()

        scalar_tags = event_acc.Tags().get("scalars", [])
        for tag in scalar_tags:
            if "train_epoch" not in tag and "val_epoch" not in tag:
                continue

            rows: list[dict[str, int | float]] = []
            for event in event_acc.Scalars(tag):
                rows.append(
                    {
                        "step": int(event.step),
                        "value": float(event.value),
                        "timestamp_ms": int(event.wall_time * 1000),
                    }
                )

            if rows:
                metric_name = tag.replace("/", "_")
                self._logger.info(
                    "[MLFLOW_TENSORBOARD_EXPORT_METRIC]",
                    f"run_id={run_id}",
                    f"metric_name={metric_name}",
                    f"points={len(rows)}",
                )
                self._mlflow_tracking.log_metric_series(
                    run_id=run_id,
                    metric_name=metric_name,
                    rows=rows,
                )

    def log_tag(self, 
            git_commit: str,
            host_name: str, 
            gpu_name: str, 
            framework: str, 
            goal: str,
            tracking_level: str,
            run_id: str
        ) -> None:
        tags = MlflowRunTags(
            source="ml/training/scripts/train_recognizer.sh",
            goal=goal,
            tracking_level=tracking_level,
            model_type="recognizer",
            task="image_classification",
            registered_model_name="deepseek vit ctc recognizer",
            git_commit=git_commit,
            host_name=host_name,
            gpu_name=gpu_name,
            framework=framework
        )
        self._mlflow_tracking.resume_run(run_id=run_id)
        self._mlflow_tracking.set_tags(
            tags.__dict__
        )
        self._mlflow_tracking.end_run(run_id=run_id)

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._mlflow_tracking.log_metrics(run_id=run_id, metrics={key: value})
