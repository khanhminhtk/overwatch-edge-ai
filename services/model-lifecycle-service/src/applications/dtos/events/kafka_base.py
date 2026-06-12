from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


KafkaEventStatus = Literal["RECEIVED", "PROCESSING", "PROCESSED", "FAILED"]


class BaseRequestedEvent(BaseModel):
    event_type: str
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    payload: Any | None = None


class BaseCompletedEvent(BaseModel):
    event_type: str
    request_id: str
    model_version: str
    artifact_uri: str
    created_at: datetime = Field(default_factory=_utcnow)
    payload: Any | None = None


class BaseFailedEvent(BaseModel):
    event_type: str
    request_id: str
    error_message: str
    created_at: datetime = Field(default_factory=_utcnow)
    payload: Any | None = None

class DeadLetterEvent(BaseModel):
    event_type: str
    dead_letter_id: str = Field(default_factory=lambda: str(uuid4()))
    original_topic: str
    original_partition: int | None = None
    original_offset: int | None = None
    error_message: str
    payload: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class BaseKafkaEvent(BaseModel):
    request_id: str
    topic: str
    partition_id: int
    message_offset: int
    consumer_group: str
    message_key: str | None = None
    event_type: str
    schema_name: str | None = None
    schema_version: str | None = None
    payload: dict[str, Any] | None = None
    status: KafkaEventStatus = "RECEIVED"
    error_message: str | None = None
    produced_at: datetime | None = None
    consumed_at: datetime = Field(default_factory=_utcnow)
    processed_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
