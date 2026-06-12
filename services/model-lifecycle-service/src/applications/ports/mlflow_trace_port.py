from __future__ import annotations

from typing import Any, Protocol


class MlflowTracePort(Protocol):
    def log_trace(
        self,
        name: str,
        request: Any | None = None,
        response: Any | None = None,
        intermediate_outputs: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        start_time_ms: int | None = None,
        execution_time_ms: int | None = None,
    ) -> str: ...

    def log_trace_metric(self, run_id: str, key: str, value: float) -> None: ...
