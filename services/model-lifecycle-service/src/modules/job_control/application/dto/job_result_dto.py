from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobResultDto:
    request_id: str
    success: bool
    error_message: str | None = None
