from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.client import initialize_mlflow_client
from src.platform.tracking.mlflow.protocols import ArtifactStore


class MlflowArtifact(ArtifactStore):
    def __init__(
        self,
        client: MlflowClient | None = None,
        tracking_url: str | None = None,
    ) -> None:
        self._client, _ = initialize_mlflow_client(
            client=client,
            tracking_url=tracking_url,
        )

    def log_artifact(
        self, run_id: str, local_path: str, artifact_path: Any = None
    ) -> None:
        self._client.log_artifact(run_id, local_path, artifact_path)

    def log_artifacts(
        self, run_id: str, local_dir: str, artifact_path: Any = None
    ) -> None:
        self._client.log_artifacts(run_id, local_dir, artifact_path)

    def log_text(
        self, run_id: str, content: str, artifact_file: str
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / Path(artifact_file).name
            file_path.write_text(content, encoding="utf-8")
            self._client.log_artifact(
                run_id,
                str(file_path),
                Path(artifact_file).parent.as_posix(),
            )

    def log_json(
        self, run_id: str, payload: dict[str, Any], artifact_file: str
    ) -> None:
        self.log_text(run_id, json.dumps(payload, indent=2), artifact_file)

    def log_external_artifact_uri(
        self, run_id: str, key: str, uri: str
    ) -> None:
        self._client.set_tag(run_id, f"artifact_uri.{key}", uri)
        self.log_json(
            run_id, {"key": key, "uri": uri}, f"external_artifacts/{key}.json"
        )
