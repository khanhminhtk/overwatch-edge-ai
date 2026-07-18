from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.modules.lifecycle.domain import ModelType


@dataclass(frozen=True, slots=True)
class GoldDatasetArtifact:
    root: Path
    archive_path: Path
    manifest_path: Path


class MlflowGoldDatasetRepository:
    """Resolves only a gold-dataset artifact tagged with exact model type/version."""

    def __init__(self, client: MlflowClient, workspace_root: Path) -> None:
        self._client = client
        self._workspace_root = workspace_root

    def download(self, model_type: ModelType, dataset_version: str) -> GoldDatasetArtifact | None:
        experiment = self._client.get_experiment_by_name(model_type.value)
        if experiment is None:
            return None
        query = f"tags.artifact_kind = 'gold_dataset' and tags.dataset_version = '{dataset_version}' and tags.model_type = '{model_type.value}'"
        runs = self._client.search_runs([experiment.experiment_id], filter_string=query, order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            return None
        destination = self._workspace_root / model_type.value / dataset_version
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        run_id = runs[0].info.run_id
        archive = Path(self._client.download_artifacts(run_id, "dataset/archive.zip", str(destination)))
        manifest = Path(self._client.download_artifacts(run_id, "dataset/manifest.json", str(destination)))
        if not archive.is_file() or not manifest.is_file():
            raise FileNotFoundError("MLflow gold dataset is missing archive.zip or manifest.json")
        return GoldDatasetArtifact(root=destination, archive_path=archive, manifest_path=manifest)
