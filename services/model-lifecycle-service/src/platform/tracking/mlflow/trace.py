from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.client import initialize_mlflow_client
from src.platform.tracking.mlflow.protocols import TracePort


class MlflowTrace(TracePort):
    def __init__(
        self,
        client: MlflowClient | None = None,
        tracking_url: str | None = None,
    ) -> None:
        self._client, _ = initialize_mlflow_client(
            client=client,
            tracking_url=tracking_url,
        )

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
