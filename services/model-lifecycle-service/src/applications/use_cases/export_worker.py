from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql

from src.applications.dtos.events.events_kafka_export import (
    ExportCompletedEvent,
    ExportFailedEvent,
    PayloadExportJob,
)
from src.applications.use_cases.job_repository import BaseJobRepository
from src.applications.use_cases.worker_process_bashscript import BaseWorkerBashScript
from src.infra.queries.kafka_event_queries import CLAIM_NEXT_PENDING_JOB
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient

RETURNING_CLAIM_NEXT_PENDING_JOB_EXPORT = sql.SQL("""
    e.id,
    e.request_id,
    e.payload->>'model_name' AS model_name,
    e.payload->>'model_version' AS model_version,
    e.status
""")


class ExportWorker(BaseWorkerBashScript):
    def __init__(
        self,
        job_repository: BaseJobRepository,
        producer: "KafkaProducerClient",
        logger: Logger,
        server_id: str,
        event_topic: str,
        training_mode: str = "local",
        idle_sleep_seconds: float = 2.0,
    ) -> None:
        super().__init__(
            job_repository=job_repository,
            producer=producer,
            logger=logger,
            server_id=server_id,
            event_topic=event_topic,
            training_mode=training_mode,
            idle_sleep_seconds=idle_sleep_seconds,
        )

    def _get_next_job(self) -> dict[str, str] | None:
        query = CLAIM_NEXT_PENDING_JOB.format(
            event_type=sql.Literal("export_requested"),
            additional_conditions=sql.SQL(""),
            returning=RETURNING_CLAIM_NEXT_PENDING_JOB_EXPORT,
        )
        return self._job_repository.claim_next_pending_job(
            server_id=self._server_id,
            query=query,
            return_params=[
                "id",
                "request_id",
                "model_name",
                "model_version",
                "status",
            ],
        )

    def _build_completed_event(self, job: dict[str, str], result: dict[str, str]) -> ExportCompletedEvent:
        return ExportCompletedEvent(
            request_id=job["request_id"],
            payload=PayloadExportJob(
                model_name=job["model_name"],
                model_version=result["model_version"],
            ),
            model_version=result["model_version"],
            artifact_uri=result["artifact_uri"],
        )

    def _build_failed_event(self, job: dict[str, str], error_message: str) -> ExportFailedEvent:
        return ExportFailedEvent(
            request_id=job["request_id"],
            error_message=error_message,
            payload=PayloadExportJob(
                model_name=job["model_name"],
                model_version=job["model_version"],
            ),
        )

    @staticmethod
    def _get_publish_key(job: dict[str, str]) -> str:
        return job["model_name"]

    def _build_job_result(self, job: dict[str, str]) -> dict[str, str]:
        model_name = job["model_name"]
        model_version = job["model_version"]
        return {
            "model_version": model_version,
            "artifact_uri": f"s3://artifacts/{model_name}/{model_version}",
        }

    def _resolve_script_name(self, job: dict[str, str]) -> str:
        model_name = job["model_name"]
        if model_name == "recognizer":
            return "export_recognizer_tensorrt.sh"
        if model_name == "detector":
            return "export_yolo_tensorrt.sh"
        raise ValueError(f"Unsupported model_name: {model_name}")
