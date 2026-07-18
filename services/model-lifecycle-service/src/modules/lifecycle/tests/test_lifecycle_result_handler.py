from src.modules.lifecycle.adapters.inbound.lifecycle_result_handler import LifecycleResultHandler
from src.modules.lifecycle.application import LifecycleCommand, LifecycleWorkflow


class Publisher:
    def __init__(self) -> None:
        self.commands: list[LifecycleCommand] = []

    def publish(self, command: LifecycleCommand) -> None:
        self.commands.append(command)


def test_dataset_result_emits_training_command() -> None:
    publisher = Publisher()
    handler = LifecycleResultHandler(LifecycleWorkflow(), publisher)
    event = {
        "status": "PROCESSED", "job_type": "dataset",
        "payload": {"lifecycle_id": "l1", "model_type": "detection", "dataset_version": "v1", "raw_data_path": "/raw"},
        "result": {"dataset_root": "/god", "archive_path": "/god.zip", "manifest_path": "/god.json", "training_data_path": "/train/v1"},
    }
    command = handler.handle_result(event)
    assert isinstance(command, LifecycleCommand)
    assert command.event_type == "train_requested_detection"
    assert command.payload["training_data_path"] == "/train/v1"
