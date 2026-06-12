from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.applications.ports.mlflow_trace_port import MlflowTracePort


class MlflowTrace(MlflowTracePort):
    def __init__(self, tracking_uri: str):
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

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
    ) -> str:
        return mlflow.log_trace(
            name=name,
            request=request,
            response=response,
            intermediate_outputs=intermediate_outputs,
            attributes=attributes,
            tags=tags,
            start_time_ms=start_time_ms,
            execution_time_ms=execution_time_ms,
        )

    def log_trace_metric(self, run_id: str, key: str, value: float) -> None:
        self._client.log_metric(run_id=run_id, key=key, value=float(value))
