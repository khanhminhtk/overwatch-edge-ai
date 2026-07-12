from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.client import initialize_mlflow_client
from src.platform.tracking.mlflow.download_resolver import (
    MlflowArtifactDownloadResolver,
)
from src.platform.tracking.mlflow.protocols import ModelRegistry


if not hasattr(mlflow, "set_tracking_url"):
    mlflow.set_tracking_url = mlflow.set_tracking_uri


class MlflowRegistry(ModelRegistry):
    _TRACKING_REACHABILITY_TIMEOUT_SECONDS = 3.0

    def __init__(
        self,
        client: MlflowClient | None = None,
        tracking_url: str | None = None,
        download_resolver: MlflowArtifactDownloadResolver | None = None,
    ) -> None:
        self._client, resolved_tracking_uri = initialize_mlflow_client(
            client=client,
            tracking_url=tracking_url,
        )
        self._tracking_uri = resolved_tracking_uri or ""
        self._download_resolver = download_resolver or MlflowArtifactDownloadResolver(
            self._tracking_uri
        )

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
        active_run = mlflow.active_run()
        if active_run is not None and active_run.info.run_id == run_id:
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

    def register_model(
        self, model_uri: str, registered_model_name: str, run_id: str
    ) -> str:
        if model_uri.startswith("runs:/"):
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

        model_version = mlflow.register_model(
            model_uri=model_uri, name=registered_model_name
        )
        return str(model_version.version)

    def set_model_version_tags(
        self,
        registered_model_name: str,
        model_version: str,
        tags: dict[str, str],
    ) -> None:
        for key, value in tags.items():
            self._client.set_model_version_tag(
                registered_model_name, model_version, key, value
            )

    def set_model_alias(
        self, registered_model_name: str, alias: str, model_version: str
    ) -> None:
        self._client.set_registered_model_alias(
            registered_model_name, alias, model_version
        )

    def get_model_version_by_alias(
        self,
        registered_model_name: str,
        alias: str,
    ) -> Any | None:
        self._ensure_tracking_server_reachable()
        try:
            return self._client.get_model_version_by_alias(
                registered_model_name,
                alias,
            )
        except MlflowException:
            return None

    def get_model_version(
        self,
        registered_model_name: str,
        version: str,
    ) -> Any:
        self._ensure_tracking_server_reachable()
        return self._client.get_model_version(
            registered_model_name,
            version,
        )

    def transition_model_stage(
        self,
        registered_model_name: str,
        model_version: str,
        stage: str,
    ) -> None:
        self._client.transition_model_version_stage(
            registered_model_name, model_version, stage
        )

    def export_model_download_url(
        self,
        model_name: str,
        version: str,
        artifact_path: str | None = None,
    ) -> str:
        self._ensure_tracking_server_reachable()
        return self._download_resolver.export_model_download_url(
            client=self._client,
            model_name=model_name,
            version=version,
            artifact_path=artifact_path,
        )

    def download_model_artifact(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        output_path: str,
    ) -> str:
        self._ensure_tracking_server_reachable()
        return self._download_resolver.download_model_artifact(
            client=self._client,
            model_name=model_name,
            version=version,
            artifact_path=artifact_path,
            output_path=output_path,
        )

    @property
    def client(self) -> MlflowClient:
        return self._client

    def _ensure_tracking_server_reachable(self) -> None:
        parsed = urllib.parse.urlparse(self._tracking_uri)
        if parsed.scheme not in {"http", "https"}:
            return

        request = urllib.request.Request(self._tracking_uri)
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._TRACKING_REACHABILITY_TIMEOUT_SECONDS,
            ):
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError(
                "MLflow tracking server is unreachable: "
                f"{self._tracking_uri}"
            ) from exc


# if __name__ == "__main__":
#     from mlflow.tracking import MlflowClient

#     from src.platform.config import ConfigLoader
#     from src.platform.tracking.mlflow.config import MlflowConfig

#     config = ConfigLoader.load(
#         MlflowConfig,
#         env_files=["services/model-lifecycle-service/config/.env"],
#         yaml_files=["services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         section="mlflow",
#     )

#     client = MlflowClient(tracking_uri=config.tracking_uri)

#     registry = MlflowRegistry(client=client)
#     print(f"Registry ready (tracking: {config.tracking_uri})")
