from src.modules.lifecycle.application import LifecycleWorkflow
from src.modules.lifecycle.domain import LifecycleContext, LifecycleStage, ModelType


def _context() -> LifecycleContext:
    return LifecycleContext(
        lifecycle_id="life-1",
        model_type=ModelType.DETECTION,
        dataset_version="v7",
        raw_data_path="/raw/detection",
    )


def test_start_uses_stage_specific_id_and_serializable_context() -> None:
    command = LifecycleWorkflow().start(_context())
    assert command.request_id == "life-1:lookup"
    assert command.event_type == "dataset_lookup_requested"
    assert command.payload["model_type"] == "detection"


def test_dataset_transition_requires_lineage_artifacts() -> None:
    command = LifecycleWorkflow().after_dataset(
        _context(),
        {"dataset_root": "/god", "archive_path": "/god.zip", "manifest_path": "/god.json"},
    )
    assert command.request_id == "life-1:training"
    assert command.event_type == "train_requested_detection"
    assert command.payload["dataset_manifest_path"] == "/god.json"


def test_context_round_trip() -> None:
    context = _context().advance(LifecycleStage.TRACKING, onnx_path="/out/model.onnx")
    assert LifecycleContext.from_payload(context.to_payload()) == context
