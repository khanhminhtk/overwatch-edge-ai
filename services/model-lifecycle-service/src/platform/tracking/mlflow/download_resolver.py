from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient


class MlflowArtifactDownloadResolver:
    def __init__(self, tracking_uri: str) -> None:
        self._tracking_uri = tracking_uri

    def export_model_download_url(
        self,
        client: MlflowClient,
        model_name: str,
        version: str,
        artifact_path: str | None = None,
    ) -> str:
        download_uri = client.get_model_version_download_uri(model_name, version)
        if download_uri.startswith("runs:/"):
            download_uri = RunsArtifactRepository.get_underlying_uri(
                download_uri,
                tracking_uri=self._tracking_uri,
            )
        if not download_uri.startswith("mlflow-artifacts:/"):
            return download_uri

        resolved_url = MlflowArtifactsRepository.resolve_uri(
            download_uri, self._tracking_uri
        )
        if artifact_path:
            return f"{resolved_url.rstrip('/')}/{artifact_path.lstrip('/')}"

        primary_path = self._resolve_primary_artifact_path(download_uri)
        if primary_path:
            return f"{resolved_url.rstrip('/')}/{primary_path.lstrip('/')}"

        return resolved_url

    def download_model_artifact(
        self,
        client: MlflowClient,
        model_name: str,
        version: str,
        artifact_path: str,
        output_path: str,
    ) -> str:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        model_version = client.get_model_version(model_name, version)
        run_id = getattr(model_version, "run_id", None)
        if not run_id:
            raise ValueError(
                f"Model version {model_name}:{version} does not have run_id"
            )

        downloaded_path = Path(
            client.download_artifacts(
                run_id=run_id,
                path=artifact_path,
                dst_path=str(destination.parent),
            )
        )
        if downloaded_path.resolve() != destination:
            downloaded_path.replace(destination)
        return str(destination)

    def _resolve_primary_artifact_path(self, artifact_uri: str) -> str | None:
        parsed = urllib.parse.urlparse(artifact_uri)
        path = parsed.path.lstrip("/")
        listing_url = (
            f"{self._tracking_uri.rstrip('/')}"
            f"/api/2.0/mlflow-artifacts/artifacts"
            f"?path={urllib.parse.quote(path)}"
        )
        try:
            req = urllib.request.Request(listing_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                listing = json.loads(resp.read().decode())
        except Exception:
            return None

        has_mlmodel = any(
            file_info.get("path") == "MLmodel"
            for file_info in listing.get("files", [])
        )
        if not has_mlmodel:
            return None

        mlmodel_url = (
            f"{self._tracking_uri.rstrip('/')}"
            f"/api/2.0/mlflow-artifacts/artifacts"
            f"/{path}/MLmodel"
        )
        try:
            req = urllib.request.Request(mlmodel_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                mlmodel = yaml.safe_load(resp.read().decode())
        except Exception:
            return None

        if not isinstance(mlmodel, dict):
            return None

        flavors = mlmodel.get("flavors", {})
        if not isinstance(flavors, dict):
            return None

        pyfunc = flavors.get("python_function", {})
        if isinstance(pyfunc, dict):
            model_file = (
                pyfunc.get("python_model")
                or pyfunc.get("model")
                or pyfunc.get("data")
            )
            if model_file:
                return str(model_file)

        for flavor in flavors.values():
            if isinstance(flavor, dict):
                for key in ("model", "data"):
                    if key in flavor:
                        return str(flavor[key])

        return None
