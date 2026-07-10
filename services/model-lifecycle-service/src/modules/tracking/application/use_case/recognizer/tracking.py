from __future__ import annotations

from pathlib import Path

from src.modules.tracking.domain.value_objects import RecognizerConfig, HardwareInfo
from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.application.use_case.tracking_common import (
    build_common_run_tags,
    build_run_name,
    get_commit_hash,
    log_exceptions,
    resolve_path,
)
from src.platform.tracking.mlflow import ExperimentTracker, TracePort
from src.platform.logger import Logger


class RecognizerMlflowTracking:
    def __init__(
        self,
        run: RecognizerRun,
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

    @log_exceptions("[RECOGNIZER_TRACKING_START_RUN_ERROR]")
    def start_run(self) -> str:
        git_commit = get_commit_hash(self._logger)
        run_id = self._experiment_tracker.start_run(
            experiment_name=self._run.experiment_name,
            run_name=build_run_name(self._run, git_commit),
        )
        self._run.run_id = run_id
        self._logger.info(f"Started MLflow run with ID: {run_id}")
        return run_id

    @log_exceptions("[RECOGNIZER_TRACKING_EXPORT_METRICS_ERROR]")
    def export_training_metrics(self, run_id: str, pwd: str) -> None:
        tensorboard_dir = self._run.config.tensorboard_dir
        if not tensorboard_dir:
            self._logger.warning("tensorboard_dir is not configured, skipping metric export.")
            return

        tb_path = resolve_path(pwd, tensorboard_dir)
        if not Path(tb_path).exists():
            self._logger.warning(
                "tensorboard_dir does not exist, skipping metric export.",
                f"path={tb_path}",
            )
            return

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self._logger.info(
            "[MLFLOW_RECOGNIZER_TENSORBOARD_EXPORT_STARTED]",
            f"run_id={run_id}",
            f"tb_file={tensorboard_dir}",
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
                    "[MLFLOW_RECOGNIZER_TENSORBOARD_EXPORT_METRIC]",
                    f"run_id={run_id}",
                    f"metric_name={metric_name}",
                    f"points={len(rows)}",
                )
                self._experiment_tracker.log_metric_series(
                    run_id=run_id,
                    metric_name=metric_name,
                    rows=rows,
                )

    @log_exceptions("[RECOGNIZER_TRACKING_LOG_METRIC_ERROR]")
    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._experiment_tracker.log_metrics(run_id=run_id, metrics={key: value})

    @log_exceptions("[RECOGNIZER_TRACKING_LOG_TAG_ERROR]")
    def log_tag(
        self,
        run_id: str,
        git_commit: str,
        source: str = "services/model-lifecycle-service/src/modules/tracking/tests/integration/recognizer_main.py",
    ) -> None:
        tags = build_common_run_tags(self._run.config, git_commit=git_commit, source=source)
        self._experiment_tracker.set_tags(run_id=run_id, tags=tags)

    @log_exceptions("[RECOGNIZER_TRACKING_EXECUTE_ERROR]")
    def execute(self, run_id: str, pwd: str) -> None:
        self._experiment_tracker.resume_run(run_id)
        self.log_tag(run_id=run_id, git_commit=get_commit_hash(self._logger))
        self.export_training_metrics(run_id, pwd=pwd)

    @log_exceptions("[RECOGNIZER_TRACKING_END_RUN_ERROR]")
    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self._experiment_tracker.end_run(run_id, status=status)

    @log_exceptions("[RECOGNIZER_TRACKING_TRACK_ERROR]")
    def track_recognizer(self, recognizer_id: str, event: str, data: dict, pwd: str) -> None:
        self._logger.info(
            f"Tracking recognizer event: recognizer_id={recognizer_id}, event={event}, data={data}"
        )
        run_id = self.start_run()
        self.execute(run_id=run_id, pwd=pwd)
        self.end_run(run_id)
