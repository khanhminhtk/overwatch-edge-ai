from __future__ import annotations

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
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
        if model_uri.startswith("runs:/"):
            run_id = model_uri.removeprefix("runs:/").split("/", 1)[0]
            try:
                self._client.create_registered_model(registered_model_name)
            except MlflowException as exc:
                if exc.error_code not in (
                    ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
                    ErrorCode.Name(ALREADY_EXISTS),
                ):
                    raise
            model_version = self._client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=run_id,
            )
            return str(model_version.version)

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

    def export_model_download_url(self, model_name: str, version: str) -> str:
        download_uri = self._client.get_model_version_download_uri(model_name, version)
        if download_uri.startswith("runs:/"):
            download_uri = RunsArtifactRepository.get_underlying_uri(
                download_uri,
                tracking_uri=self._tracking_uri,
            )
        if download_uri.startswith("mlflow-artifacts:/"):
            return MlflowArtifactsRepository.resolve_uri(download_uri, self._tracking_uri)
        return download_uri

    @property
    def client(self) -> MlflowClient:
        return self._client
