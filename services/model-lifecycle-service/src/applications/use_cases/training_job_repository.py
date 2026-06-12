from __future__ import annotations

from typing import Any

from psycopg import sql

from src.applications.use_cases.job_repository import BaseJobRepository
from src.infra.queries.kafka_event_queries import CLAIM_NEXT_PENDING_TRAINING_JOB
from src.infra.sql import PostgresSQLHandler
from src.utils.logger import Logger


class TrainingJobRepository(BaseJobRepository):
    def __init__(self, sql_handler: PostgresSQLHandler, logger: Logger) -> None:
        super().__init__(
            sql_handler=sql_handler,
            logger=logger,
            schema_name="training_event",
        )

    def claim_next_pending_job(
        self,
        server_id: str,
        query: sql.SQL | str | None = None,
        return_params: list[str] | None = None,
    ) -> dict[str, Any] | None:
        return super().claim_next_pending_job(
            server_id=server_id,
            query=query or CLAIM_NEXT_PENDING_TRAINING_JOB,
            return_params=return_params or [
                "id",
                "request_id",
                "model_name",
                "dataset_version",
                "train_config_uri",
                "target_server_id",
                "status",
            ],
        )

    def get_claim_next_pending_job(self, server_id: str):
        return self.claim_next_pending_job(server_id)
