from __future__ import annotations

from mlflow.entities import Run
from mlflow.tracking import MlflowClient


def assert_run_created(client: MlflowClient, experiment_name: str, run_id: str) -> Run:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise AssertionError(f"Experiment not found: {experiment_name}")

    run = client.get_run(run_id)
    if run.info.experiment_id != experiment.experiment_id:
        raise AssertionError(
            f"Run {run_id} belongs to experiment {run.info.experiment_id}, expected {experiment.experiment_id}"
        )
    return run


def delete_experiment(client: MlflowClient, experiment_name: str) -> bool:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False
    client.delete_experiment(experiment.experiment_id)
    return True
