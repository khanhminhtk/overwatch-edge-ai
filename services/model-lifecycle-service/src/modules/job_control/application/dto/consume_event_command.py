from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity


@dataclass(frozen=True)
class ConsumeEventCommand:
    request_id: str
    event_type: str
    schema_name: str | None
    schema_version: str | None
    message_key: str | None
    message_identity: MessageIdentity
    payload: dict[str, Any] | None
