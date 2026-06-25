import sys
from pathlib import Path
import os

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.tracking import BaseMlflowTrackingUseCase
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger

class MlflowTrackingRecognizer(BaseMlflowTrackingUseCase):
    def __init__(
        self,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str
    ):
        super().__init__(
            mlflow_tracking=mlflow_tracking,
            logger=logger,
            path_env_ml_traning=path_env_ml_traning,
            model_name_env_key="MLFLOW_MODEL_NAME_RECOGNIZER",
            model_type_env_key="RECOG_MODEL_TYPE",
            task_name_env_key="RECOG_TASK_NAME",
            docker_image_env_key="RECOG_DOCKER_IMAGE",
            dataset_zip_path_env_key="RECOG_DATASET_ZIP_PATH",
            manifest_path_env_key="RECOG_MANIFEST_PATH",
        )
        self._tensorboard_dir = self._env("RECOG_TENSORBOARD_DIR")
        self._checkpoint_dir = self._env("RECOG_SAVE_DIR")

    def export_training_metrics(self, run_id: str, pwd: str) -> None:
        if not self._tensorboard_dir:
            raise ValueError("RECOG_TENSORBOARD_FILE is not configured")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tb_path = self._resolve_path(pwd, self._tensorboard_dir)

        self._logger.info(
            "[MLFLOW_TENSORBOARD_EXPORT_STARTED]",
            f"run_id={run_id}",
            f"tb_file={self._tensorboard_dir}",
        )

        event_acc = EventAccumulator(str(tb_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags().get("scalars", [])
        for tag in scalar_tags:
            if "train/epoch" not in tag and "val/epoch" not in tag:
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

    def _metric_source_tag_value(self) -> str:
        return self._tensorboard_dir

    def _checkpoint_dir_tag_value(self) -> str:
        return self._checkpoint_dir

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._mlflow_tracking.log_metrics(run_id=run_id, metrics={key: value})


# if __name__ == "__main__":
#     from src.infra.mlflow.mlflow_tracking import MlflowTracking
#     from src.utils.logger import Logger
#     print(SERVICE_ROOT)

#     mlflow_tracking = MlflowTracking("http://localhost:5000")

#     logger = Logger()
#     recognizer = MlflowTrackingRecognizer(
#         mlflow_tracking=mlflow_tracking,
#         logger=logger,
#         path_env_ml_traning="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/config/.env.example"
#     )
#     recognizer.excute(
#         git_commit="abc123",
#         host_name="test_host",
#         gpu_name="test_gpu",
#         framework="pytorch",
#         goal="test_goal",
#         tracking_level="test_level"
#     )
