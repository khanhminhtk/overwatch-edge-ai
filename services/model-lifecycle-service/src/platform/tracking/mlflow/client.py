from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


def initialize_mlflow_client(
    *,
    client: MlflowClient | None = None,
    tracking_url: str | None = None,
) -> tuple[MlflowClient, str | None]:
    if client is None and tracking_url is None:
        raise ValueError("Either client or tracking_url must be provided")

    resolved_tracking_uri = tracking_url or getattr(client, "tracking_uri", None)
    if resolved_tracking_uri:
        mlflow.set_tracking_uri(resolved_tracking_uri)

    if client is None:
        client = MlflowClient(tracking_uri=resolved_tracking_uri)

    return client, resolved_tracking_uri
