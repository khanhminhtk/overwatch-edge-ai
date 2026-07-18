from __future__ import annotations

import shutil
from pathlib import Path

from src.modules.lifecycle.domain import LifecycleContext


class DatasetWorkspaceMaterializer:
    """Copies a verified dataset into a versioned training workspace without overwrite."""

    def __init__(self, training_data_root: Path) -> None:
        self._training_data_root = training_data_root.resolve()

    def materialize(self, context: LifecycleContext) -> LifecycleContext:
        if not context.dataset_root:
            raise ValueError("dataset_root is required before materializing training data")
        source = Path(context.dataset_root).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"dataset root not found: {source}")
        self._validate(source, context.model_type.value)
        destination = self._training_data_root / context.model_type.value / context.dataset_version
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return context.advance(context.stage, training_data_path=str(destination))

    @staticmethod
    def _validate(root: Path, model_type: str) -> None:
        required = (("train", "images"), ("train", "labels")) if model_type == "detection" else (("train_data",),)
        if any(not root.joinpath(*parts).exists() for parts in required):
            raise ValueError(f"invalid {model_type} dataset layout at {root}")
