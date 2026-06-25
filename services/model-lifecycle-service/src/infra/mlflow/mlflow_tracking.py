from __future__ import annotations

from typing import Any, Mapping, Sequence

import mlflow
from mlflow.tracking import MlflowClient

from src.applications.ports.mlflow_tracking_port import MlflowTrackingPort


class MlflowTracking(MlflowTrackingPort):
    def __init__(self, tracking_uri: str):
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def ensure_experiment(self, experiment_name: str) -> str:
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return self._client.create_experiment(experiment_name)

    def start_run(self, experiment_name: str, run_name: str) -> str:
        experiment_id = self.ensure_experiment(experiment_name)
        run = self._client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={"mlflow.runName": run_name},
        )
        return run.info.run_id

    def resume_run(self, run_id: str) -> str:
        self._client.get_run(run_id)
        return run_id

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self._client.set_terminated(run_id, status=status)

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        for key, value in tags.items():
            self._client.set_tag(run_id, key, str(value))

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self._client.log_param(run_id, key, str(value))

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, float(value))

    def log_metric_series(
        self,
        run_id: str,
        metric_name: str,
        rows: Sequence[Mapping[str, int | float]],
    ) -> None:
        for row in rows:
            self._client.log_metric(
                run_id=run_id,
                key=metric_name,
                value=float(row["value"]),
                step=int(row.get("step", 0)),
                timestamp=int(row.get("timestamp_ms", 0)) if "timestamp_ms" in row else None,
            )

    @property
    def client(self) -> MlflowClient:
        return self._client
