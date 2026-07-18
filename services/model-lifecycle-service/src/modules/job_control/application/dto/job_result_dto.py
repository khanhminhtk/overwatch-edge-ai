from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JobResultDto:
    request_id: str
    success: bool
    error_message: str | None = None
    result: dict[str, Any] | None = None
