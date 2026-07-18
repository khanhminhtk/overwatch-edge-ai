from __future__ import annotations

from typing import Any, Protocol

from src.modules.lifecycle.application import LifecycleCommand, LifecycleWorkflow
from src.modules.lifecycle.domain import LifecycleContext, LifecycleStage


class CommandPublisher(Protocol):
    def publish(self, command: LifecycleCommand) -> None: ...


class LifecycleResultHandler:
    """Consumes successful worker results and emits exactly the next lifecycle command."""

    def __init__(self, workflow: LifecycleWorkflow, publisher: CommandPublisher) -> None:
        self._workflow = workflow
        self._publisher = publisher

    def start(self, payload: dict[str, Any]) -> LifecycleCommand:
        command = self._workflow.start(LifecycleContext.from_payload(payload))
        self._publisher.publish(command)
        return command

    def handle_result(self, event: dict[str, Any]) -> LifecycleCommand | LifecycleContext | None:
        if event.get("status") != "PROCESSED":
            return None
        payload = event.get("payload")
        result = event.get("result")
        if not isinstance(payload, dict) or not isinstance(result, dict):
            raise ValueError("lifecycle result event requires object payload and result")
        context = LifecycleContext.from_payload(payload)
        job_type = event.get("job_type")
        if job_type == "dataset":
            command = self._workflow.after_dataset(context, result)
        elif job_type == "training":
            command = self._workflow.after_training(context, result)
        elif job_type == "export_onnx":
            command = self._workflow.after_export(context, result)
        elif job_type == "mlflow_tracking":
            return self._workflow.complete(context, result)
        else:
            return None
        self._publisher.publish(command)
        return command
