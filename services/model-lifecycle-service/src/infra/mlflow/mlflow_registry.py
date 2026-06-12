from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.applications.ports.mlflow_model_registry_port import MlflowModelRegistryPort


class MlflowRegistry(MlflowModelRegistryPort):
    def __init__(self, tracking_uri: str):
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def log_pyfunc_model(
        self,
        run_id: str,
        model_name: str,
        python_model: Any,
        input_example: Any,
        signature: Any,
        registered_model_name: str | None = None,
        tags: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        with mlflow.start_run(run_id=run_id):
            model_info = mlflow.pyfunc.log_model(
                name=model_name,
                python_model=python_model,
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
                tags=tags,
                params=params,
                model_type="pyfunc",
            )
        return model_info.model_uri

    def register_model(self, model_uri: str, registered_model_name: str) -> str:
        model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        return str(model_version.version)

    def set_model_version_tags(
        self,
        registered_model_name: str,
        model_version: str,
        tags: dict[str, str],
    ) -> None:
        for key, value in tags.items():
            self._client.set_model_version_tag(registered_model_name, model_version, key, value)

    def set_model_alias(self, registered_model_name: str, alias: str, model_version: str) -> None:
        self._client.set_registered_model_alias(registered_model_name, alias, model_version)

    def transition_model_stage(
        self,
        registered_model_name: str,
        model_version: str,
        stage: str,
    ) -> None:
        self._client.transition_model_version_stage(registered_model_name, model_version, stage)
