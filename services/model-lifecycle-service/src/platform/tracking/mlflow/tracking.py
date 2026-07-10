from __future__ import annotations

from typing import Any, Mapping, Sequence

from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.client import initialize_mlflow_client
from src.platform.tracking.mlflow.protocols import ExperimentTracker


class MlflowTracking(ExperimentTracker):
    def __init__(
        self,
        client: MlflowClient | None = None,
        tracking_url: str | None = None,
    ) -> None:
        self._client, _ = initialize_mlflow_client(
            client=client,
            tracking_url=tracking_url,
        )

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
                timestamp=int(row.get("timestamp_ms", 0))
                if "timestamp_ms" in row
                else None,
            )

    @property
    def client(self) -> MlflowClient:
        return self._client


# if __name__ == "__main__":
#     from src.platform.tracking.mlflow.config import MlflowConfig
#     from src.platform.config import ConfigLoader

#     config = ConfigLoader.load(
#         MlflowConfig,
#         yaml_files=["services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files=["services/model-lifecycle-service/config/.env"],
#         section="mlflow"
#     )


#     client = MlflowClient(tracking_uri=config.tracking_uri)
#     tracking = MlflowTracking(client=client)
#     experiment_id = tracking.ensure_experiment("test-experiment")
#     print(f"Experiment ID: {experiment_id}")
