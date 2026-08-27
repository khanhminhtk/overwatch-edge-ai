from pathlib import Path
from zipfile import ZipFile

import pytest

from src.modules.lifecycle.application.fine_tuning_materializer import FineTuningMaterializer
from src.modules.lifecycle.domain import ModelType


class _Gold:
    def download(self, *_args):
        return None


class _Client:
    def __init__(self, archive: Path): self.archive = archive
    def download_artifacts(self, _run_id, _path, _dst): return str(self.archive)


def test_materialize_downloads_labeled_mlflow_archive(tmp_path: Path) -> None:
    archive = tmp_path / "labeled.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("train/images/a.jpg", b"image")
        zipped.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2")
    materializer = FineTuningMaterializer(_Client(archive), _Gold(), tmp_path / "workspace")
    result = materializer.materialize(model_type=ModelType.DETECTION, dataset_version="v1", labeled_dataset_uri="runs:/run-1/dataset/archive.zip")
    assert (result / "train/images/a.jpg").is_file()


def test_rejects_non_mlflow_dataset_uri(tmp_path: Path) -> None:
    materializer = FineTuningMaterializer(_Client(tmp_path / "x.zip"), _Gold(), tmp_path)
    with pytest.raises(ValueError, match="runs:"):
        materializer.materialize(model_type=ModelType.DETECTION, dataset_version="v1", labeled_dataset_uri="s3://bucket/data.zip")
