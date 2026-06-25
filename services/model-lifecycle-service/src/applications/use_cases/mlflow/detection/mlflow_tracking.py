from __future__ import annotations

import csv
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.tracking import BaseMlflowTrackingUseCase
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger


class MlflowTrackingDetection(BaseMlflowTrackingUseCase):
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
            model_name_env_key="MLFLOW_MODEL_NAME_DETECTION",
            model_type_env_key="DETECT_MODEL_TYPE",
            task_name_env_key="DETECT_TASK_NAME",
            docker_image_env_key="DETECT_DOCKER_IMAGE",
            dataset_zip_path_env_key="DETECT_DATASET_ZIP_PATH",
            manifest_path_env_key="DETECT_MANIFEST_PATH",
        )
        self._results_csv_path = self._env("DETECT_RESULTS_CSV_PATH")
        self._checkpoint_dir = self._env("DETECT_SAVE_DIR")

    def export_training_metrics(self, run_id: str, pwd: str) -> None:
        if not self._results_csv_path:
            raise ValueError("DETECT_RESULTS_CSV_PATH is not configured")

        results_csv_path = self._resolve_path(pwd, self._results_csv_path)
        self._logger.info(
            "[MLFLOW_DETECTION_RESULTS_EXPORT_STARTED]",
            f"run_id={run_id}",
            f"results_csv={self._results_csv_path}",
        )

        with open(results_csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
        if not rows:
            return

        series_by_metric: dict[str, list[dict[str, int | float]]] = {}
        for row in rows:
            step = int(float(row.get("epoch", "0")))
            for key, raw_value in row.items():
                if key is None or key == "epoch" or raw_value in (None, ""):
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                metric_name = key.strip().replace("/", "_").replace("(", "").replace(")", "")
                series_by_metric.setdefault(metric_name, []).append(
                    {
                        "step": step,
                        "value": value,
                    }
                )

        for metric_name, metric_rows in series_by_metric.items():
            self._logger.info(
                "[MLFLOW_DETECTION_RESULTS_EXPORT_METRIC]",
                f"run_id={run_id}",
                f"metric_name={metric_name}",
                f"points={len(metric_rows)}",
            )
            self._mlflow_tracking.log_metric_series(
                run_id=run_id,
                metric_name=metric_name,
                rows=metric_rows,
            )

    def _metric_source_tag_value(self) -> str:
        return self._results_csv_path

    def _checkpoint_dir_tag_value(self) -> str:
        return self._checkpoint_dir
