from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.modules.lifecycle.adapters.outbound.mlflow_gold_dataset import MlflowGoldDatasetRepository
from src.modules.lifecycle.domain import ModelType


class FineTuningMaterializer:
    """Builds a trainable version from MLflow labeled data and prior God Data."""

    def __init__(self, client: MlflowClient, gold_data: MlflowGoldDatasetRepository, workspace_root: Path) -> None:
        self._client = client
        self._gold_data = gold_data
        self._workspace_root = workspace_root.resolve()

    def materialize(self, *, model_type: ModelType, dataset_version: str, labeled_dataset_uri: str) -> Path:
        version = _version_number(dataset_version)
        destination = self._workspace_root / model_type.value / dataset_version
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)

        for previous in range(1, version):
            artifact = self._gold_data.download(model_type, f"v{previous}")
            if artifact is not None:
                _extract_zip(artifact.archive_path, destination)

        run_id, artifact_path = _parse_runs_uri(labeled_dataset_uri)
        archive = Path(self._client.download_artifacts(run_id, artifact_path, str(destination / ".labeled")))
        _extract_zip(archive, destination)
        shutil.rmtree(destination / ".labeled", ignore_errors=True)
        _validate_layout(destination, model_type)
        return destination


def _parse_runs_uri(uri: str) -> tuple[str, str]:
    prefix = "runs:/"
    if not uri.startswith(prefix):
        raise ValueError("labeled_dataset_uri must be MLflow runs:/<run_id>/<artifact_path>")
    parts = uri[len(prefix):].split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("labeled_dataset_uri must contain run_id and artifact path")
    return parts[0], parts[1]


def _version_number(version: str) -> int:
    if not version.startswith("v") or not version[1:].isdigit() or int(version[1:]) <= 0:
        raise ValueError("dataset_version must be v<positive integer>")
    return int(version[1:])


def _extract_zip(archive: Path, destination: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive}")
    with zipfile.ZipFile(archive) as zip_file:
        root = destination.resolve()
        for member in zip_file.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zip_file.extractall(root)


def _validate_layout(root: Path, model_type: ModelType) -> None:
    required = (("train", "images"), ("train", "labels")) if model_type is ModelType.DETECTION else (("train_data",),)
    if any(not root.joinpath(*parts).exists() for parts in required):
        raise ValueError(f"invalid {model_type.value} merged dataset layout at {root}")
