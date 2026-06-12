from __future__ import annotations

from uuid import uuid4
from typing import TYPE_CHECKING

from psycopg import sql

from src.applications.dtos.events.events_kafka_training import (
    PayloadTrainingJob,
    TrainingCompletedEvent,
    TrainingFailedEvent,
)
from src.applications.use_cases.job_repository import BaseJobRepository
from src.applications.use_cases.worker_process_bashscript import BaseWorkerBashScript
from src.infra.queries.kafka_event_queries import CLAIM_NEXT_PENDING_JOB
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient

RETURNING_CLAIM_NEXT_PENDING_JOB_TRAINING = sql.SQL("""
    e.id,
    e.request_id,
    e.payload->>'model_name' AS model_name,
    e.payload->>'dataset_version' AS dataset_version,
    e.payload->>'train_config_uri' AS train_config_uri,
    e.payload->>'target_server_id' AS target_server_id,
    e.status
""")

ADDITIONAL_CONDITIONS_CLAIM_NEXT_PENDING_JOB_TRAINING = sql.SQL("""
    AND running.payload->>'model_name' = e.payload->>'model_name'
""")


class TrainingWorker(BaseWorkerBashScript):
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
            event_type=sql.Literal("train_requested"),
            additional_conditions=ADDITIONAL_CONDITIONS_CLAIM_NEXT_PENDING_JOB_TRAINING,
            returning=RETURNING_CLAIM_NEXT_PENDING_JOB_TRAINING,
        )
        return self._job_repository.claim_next_pending_job(
            server_id=self._server_id,
            query=query,
            return_params=[
                "id",
                "request_id",
                "model_name",
                "dataset_version",
                "train_config_uri",
                "target_server_id",
                "status",
            ],
        )

    def _build_completed_event(self, job: dict[str, str], result: dict[str, str]) -> TrainingCompletedEvent:
        return TrainingCompletedEvent(
            request_id=job["request_id"],
            payload=PayloadTrainingJob(
                model_name=job["model_name"],
                dataset_version=job["dataset_version"],
                train_config_uri=job.get("train_config_uri"),
                target_server_id=job.get("target_server_id"),
                mlflow_run_id=result["mlflow_run_id"],
            ),
            model_version=result["model_version"],
            artifact_uri=result["artifact_uri"],
        )

    def _build_failed_event(self, job: dict[str, str], error_message: str) -> TrainingFailedEvent:
        return TrainingFailedEvent(
            request_id=job["request_id"],
            error_message=error_message,
            payload=PayloadTrainingJob(
                model_name=job["model_name"],
                dataset_version=job["dataset_version"],
                train_config_uri=job.get("train_config_uri"),
                target_server_id=job.get("target_server_id"),
            ),
        )

    @staticmethod
    def _get_publish_key(job: dict[str, str]) -> str:
        return job["model_name"]

    def _build_job_result(self, job: dict[str, str]) -> dict[str, str]:
        model_name = job["model_name"]
        model_version = f"{model_name}-{uuid4().hex[:8]}"
        return {
            "model_version": model_version,
            "artifact_uri": f"s3://artifacts/{model_name}/{model_version}",
            "mlflow_run_id": uuid4().hex,
        }

    def _resolve_script_name(self, job: dict[str, str]) -> str:
        model_name = job["model_name"]
        if model_name == "recognizer":
            return "train_recognizer.sh"
        if model_name == "detector":
            return "train_yolo.sh"
        raise ValueError(f"Unsupported model_name: {model_name}")
