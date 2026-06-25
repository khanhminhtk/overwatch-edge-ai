from __future__ import annotations

from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger


class BaseMlflowWorkflowUseCase:
    def __init__(
        self,
        *,
        mlflow_tracking: MlflowTracking,
        artifacts_use_case: object,
        tracking_use_case: object,
        model_registry_use_case: object,
        logger: Logger,
        experiment_name: str,
        run_name_prefix: str,
        model_name: str,
        gpu_name: str,
    ):
        self._mlflow_tracking = mlflow_tracking
        self._artifacts_use_case = artifacts_use_case
        self._tracking_use_case = tracking_use_case
        self._model_registry_use_case = model_registry_use_case
        self._logger = logger
        self._experiment_name = experiment_name
        self._run_name_prefix = run_name_prefix
        self._model_name = model_name
        self._gpu_name = gpu_name

    def execute(
        self,
        git_commit: str,
        checkpoint_name: list[str],
        goal: str,
        tracking_level: str,
        pwd: str | None = None,
    ) -> None:
        resolved_pwd = pwd or ""
        status = "FAILED"
        run_id = self._mlflow_tracking.start_run(
            experiment_name=self._experiment_name,
            run_name=f"{self._run_name_prefix}_{self._model_name}_{git_commit}_{self._gpu_name}",
        )
        self._logger.info("[MLFLOW_WORKFLOW_RUN_STARTED]", f"run_id={run_id}", f"model_name={self._model_name}")
        try:
            self._mlflow_tracking.resume_run(run_id)
            self._artifacts_use_case.execute(
                checkpoint_name=checkpoint_name,
                run_id=run_id,
                pwd=resolved_pwd,
            )
            self._logger.info("[MLFLOW_WORKFLOW_ARTIFACTS_UPLOADED]", f"run_id={run_id}")
            self._tracking_use_case.execute(
                git_commit=git_commit,
                goal=goal,
                tracking_level=tracking_level,
                run_id=run_id,
                pwd=resolved_pwd,
            )
            self._logger.info("[MLFLOW_WORKFLOW_TRACKING_LOGGED]", f"run_id={run_id}")
            self._model_registry_use_case.execute(
                run_id=run_id,
                git_commit=git_commit,
                pwd=resolved_pwd,
            )
            self._logger.info("[MLFLOW_WORKFLOW_MODEL_REGISTERED]", f"run_id={run_id}")
            status = "FINISHED"
        except Exception as exc:
            self._logger.exception(
                "[MLFLOW_WORKFLOW_RUN_FAILED]",
                f"run_id={run_id}",
                f"error={exc}",
            )
            raise
        finally:
            self._mlflow_tracking.end_run(run_id, status=status)
            self._logger.info("[MLFLOW_WORKFLOW_RUN_FINISHED]", f"run_id={run_id}", f"status={status}")

    def export_model_download_url(self, model_name: str, version: str) -> str:
        return self._model_registry_use_case.export_model_download_url(model_name, version)
