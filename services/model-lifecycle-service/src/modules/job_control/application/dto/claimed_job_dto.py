from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClaimedJobDto:
    request_id: str
    event_type: str
    payload: dict[str, Any] | None
    status: str
