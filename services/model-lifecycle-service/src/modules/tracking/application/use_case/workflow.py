from __future__ import annotations

from src.platform.logger import Logger
from src.modules.tracking.application.ports import ArtifactPort, RegistryPort, TrackingPort
from src.modules.tracking.application.use_case.tracking_common import log_exceptions


class MlflowWorkflow:
    def __init__(
        self,
        tracking: TrackingPort,
        artifacts: ArtifactPort,
        registry: RegistryPort,
        logger: Logger | None = None,
    ) -> None:
        self._tracking = tracking
        self._artifacts = artifacts
        self._registry = registry
        self._logger = logger

    @log_exceptions("[MLFLOW_WORKFLOW_EXECUTE_ERROR]")
    def execute(self, checkpoint_names: list[str], pwd: str) -> str:
        run_id = self._tracking.start_run()
        try:
            self._tracking.execute(run_id=run_id, pwd=pwd)
            self._artifacts.execute(
                checkpoint_name=checkpoint_names,
                run_id=run_id,
                pwd=pwd,
            )
            model_version = self._registry.execute(run_id=run_id, pwd=pwd)
        except Exception:
            self._tracking.end_run(run_id=run_id, status="FAILED")
            raise

        self._tracking.end_run(run_id=run_id)
        return model_version
