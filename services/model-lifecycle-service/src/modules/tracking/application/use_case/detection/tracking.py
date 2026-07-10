from __future__ import annotations

import csv

from src.modules.tracking.domain.value_objects import DetectionConfig, HardwareInfo
from src.modules.tracking.domain.entity_objects import DetectionRun
from src.modules.tracking.application.use_case.tracking_common import (
    build_common_run_tags,
    build_run_name,
    get_commit_hash,
    log_exceptions,
    resolve_path,
)
from src.platform.tracking.mlflow import ExperimentTracker, TracePort
from src.platform.logger import Logger


class DetectionMlflowTracking:
    def __init__(
        self,
        run: DetectionRun,
        hardware_info: HardwareInfo,
        experiment_tracker: ExperimentTracker,
        trace_port: TracePort,
        logger: Logger,
    ) -> None:
        self._run = run
        self._hardware_info = hardware_info
        self._experiment_tracker = experiment_tracker
        self._trace_port = trace_port
        self._logger = logger

    @log_exceptions("[DETECTION_TRACKING_START_RUN_ERROR]")
    def start_run(self) -> str:
        git_commit = get_commit_hash(self._logger)
        run_id = self._experiment_tracker.start_run(
            experiment_name=self._run.experiment_name,
            run_name=build_run_name(self._run, git_commit),
        )
        self._run.run_id = run_id
        self._logger.info(f"Started MLflow run with ID: {run_id}")
        return run_id

    @log_exceptions("[DETECTION_TRACKING_EXPORT_METRICS_ERROR]")
    def export_training_metrics(self, run_id: str, pwd: str) -> None:
        results_csv_path = self._run.config.results_csv_path
        if not results_csv_path:
            raise ValueError("DETECT_RESULTS_CSV_PATH is not configured")

        csv_path = resolve_path(pwd, results_csv_path)
        self._logger.info(
            "[MLFLOW_DETECTION_RESULTS_EXPORT_STARTED]",
            f"run_id={run_id}",
            f"results_csv={results_csv_path}",
        )

        with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
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
            self._experiment_tracker.log_metric_series(
                run_id=run_id,
                metric_name=metric_name,
                rows=metric_rows,
            )

    @log_exceptions("[DETECTION_TRACKING_LOG_METRIC_ERROR]")
    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._experiment_tracker.log_metrics(run_id=run_id, metrics={key: value})

    @log_exceptions("[DETECTION_TRACKING_LOG_TAG_ERROR]")
    def log_tag(
        self,
        run_id: str,
        git_commit: str,
        source: str = "services/model-lifecycle-service/src/modules/tracking/tests/integration/detection_main.py",
    ) -> None:
        tags = build_common_run_tags(self._run.config, git_commit=git_commit, source=source)
        self._experiment_tracker.set_tags(run_id=run_id, tags=tags)

    @log_exceptions("[DETECTION_TRACKING_EXECUTE_ERROR]")
    def execute(self, run_id: str, pwd: str) -> None:
        self._experiment_tracker.resume_run(run_id)
        self.log_tag(run_id=run_id, git_commit=get_commit_hash(self._logger))
        self.export_training_metrics(run_id, pwd=pwd)

    @log_exceptions("[DETECTION_TRACKING_END_RUN_ERROR]")
    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self._experiment_tracker.end_run(run_id, status=status)

    @log_exceptions("[DETECTION_TRACKING_TRACK_ERROR]")
    def track_detection(self, detection_id: str, event: str, data: dict, pwd: str) -> None:
        self._logger.info(
            f"Tracking detection event: detection_id={detection_id}, event={event}, data={data}"
        )
        run_id = self.start_run()
        self.execute(run_id=run_id, pwd=pwd)
        self.end_run(run_id)
