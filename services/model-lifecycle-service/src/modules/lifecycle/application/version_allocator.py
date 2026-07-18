from __future__ import annotations

from typing import Protocol

from src.modules.lifecycle.domain import LifecycleContext, ModelType


class DatasetVersionAllocator(Protocol):
    async def allocate(self, model_type: ModelType) -> str: ...


class LifecycleContextFactory:
    """Creates the first context and owns automatic dataset-version allocation."""

    def __init__(self, allocator: DatasetVersionAllocator) -> None:
        self._allocator = allocator

    async def create(
        self,
        *,
        lifecycle_id: str,
        model_type: ModelType,
        raw_data_path: str,
        dataset_version: str | None = None,
    ) -> LifecycleContext:
        version = dataset_version.strip() if dataset_version and dataset_version.strip() else await self._allocator.allocate(model_type)
        return LifecycleContext(
            lifecycle_id=lifecycle_id,
            model_type=model_type,
            dataset_version=version,
            raw_data_path=raw_data_path,
        )
