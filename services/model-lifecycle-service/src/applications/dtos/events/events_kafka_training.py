from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from src.applications.dtos.events.kafka_base import (
    BaseCompletedEvent,
    BaseFailedEvent,
    BaseKafkaEvent,
    BaseRequestedEvent,
    DeadLetterEvent,
)


class PayloadTrainingJob(BaseModel):
    version: Literal["1.0"] = "1.0"
    model_name: str
    dataset_version: str
    train_config_uri: str | None = None
    target_server_id: str | None = None
    mlflow_run_id: str | None = None

    def get_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, json_str: str) -> "PayloadTrainingJob":
        return cls(**json.loads(json_str))


class BaseTrainingEvent(BaseModel):
    payload: PayloadTrainingJob


class TrainRequestedEvent(BaseRequestedEvent, BaseTrainingEvent):
    event_type: Literal["train_requested"] = "train_requested"


class TrainingRequestedEvent(TrainRequestedEvent):
    pass


class TrainingCompletedEvent(BaseCompletedEvent, BaseTrainingEvent):
    event_type: Literal["training_completed"] = "training_completed"


class TrainingFailedEvent(BaseFailedEvent, BaseTrainingEvent):
    event_type: Literal["training_failed"] = "training_failed"


class KafkaTrainingEvent(BaseKafkaEvent):
    payload: dict[str, Any] | None = None


__all__ = [
    "BaseTrainingEvent",
    "DeadLetterEvent",
    "KafkaTrainingEvent",
    "PayloadTrainingJob",
    "TrainRequestedEvent",
    "TrainingCompletedEvent",
    "TrainingFailedEvent",
    "TrainingRequestedEvent",
]
