from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from src.applications.dtos.events.kafka_base import (
    BaseCompletedEvent,
    BaseFailedEvent,
    BaseKafkaEvent,
    BaseRequestedEvent,
)


class PayloadExportJob(BaseModel):
    version: Literal["1.0"] = "1.0"
    model_name: str
    model_version: str

    def get_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, json_str: str) -> "PayloadExportJob":
        return cls(**json.loads(json_str))


class BaseExportEvent(BaseModel):
    payload: PayloadExportJob


class ExportRequestedEvent(BaseRequestedEvent, BaseExportEvent):
    event_type: Literal["export_requested"] = "export_requested"


class ExportCompletedEvent(BaseCompletedEvent, BaseExportEvent):
    event_type: Literal["export_completed"] = "export_completed"


class ExportFailedEvent(BaseFailedEvent, BaseExportEvent):
    event_type: Literal["export_failed"] = "export_failed"


class KafkaExportEvent(BaseKafkaEvent):
    payload: dict[str, Any] | None = None


__all__ = [
    "BaseExportEvent",
    "ExportCompletedEvent",
    "ExportFailedEvent",
    "ExportRequestedEvent",
    "KafkaExportEvent",
    "PayloadExportJob",
]
