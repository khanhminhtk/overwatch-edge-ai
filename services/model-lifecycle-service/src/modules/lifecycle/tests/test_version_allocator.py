import pytest

from src.modules.lifecycle.application.version_allocator import LifecycleContextFactory
from src.modules.lifecycle.domain import ModelType


class Allocator:
    def __init__(self) -> None:
        self.calls: list[ModelType] = []

    async def allocate(self, model_type: ModelType) -> str:
        self.calls.append(model_type)
        return "v12"


@pytest.mark.asyncio
async def test_factory_allocates_when_version_is_omitted() -> None:
    allocator = Allocator()
    context = await LifecycleContextFactory(allocator).create(
        lifecycle_id="life-1", model_type=ModelType.RECOGNIZER, raw_data_path="/raw"
    )
    assert context.dataset_version == "v12"
    assert allocator.calls == [ModelType.RECOGNIZER]


@pytest.mark.asyncio
async def test_factory_preserves_explicit_version() -> None:
    allocator = Allocator()
    context = await LifecycleContextFactory(allocator).create(
        lifecycle_id="life-1", model_type=ModelType.DETECTION, raw_data_path="/raw", dataset_version="release-7"
    )
    assert context.dataset_version == "release-7"
    assert allocator.calls == []
