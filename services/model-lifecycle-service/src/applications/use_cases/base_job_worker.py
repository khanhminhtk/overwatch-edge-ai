from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.applications.use_cases.job_repository import BaseJobRepository
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient


class BaseJobWorker(ABC):
    def __init__(
        self,
        *,
        job_repository: BaseJobRepository,
        producer: "KafkaProducerClient",
        logger: Logger,
        server_id: str,
        event_topic: str,
        idle_sleep_seconds: float = 2.0,
    ) -> None:
        self._job_repository = job_repository
        self._producer = producer
        self._logger = logger
        self._server_id = server_id
        self._event_topic = event_topic
        self._idle_sleep_seconds = idle_sleep_seconds

    def run_forever(self) -> None:
        self._logger.info("[WORKER_STARTED]", f"server_id={self._server_id}")
        while True:
            processed = self.process_next_job()
            if not processed:
                time.sleep(self._idle_sleep_seconds)

    def process_next_job(self) -> bool:
        job = self._get_next_job()
        if job is None:
            return False

        request_id = self._get_request_id(job)
        self._logger.info("[JOB_RUNNING]", *self._build_running_log_fields(job))

        started_at = time.perf_counter()
        try:
            result = self._run_job(job)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            self._job_repository.make_job_successful(request_id)
            event = self._build_completed_event(job, result)
            self._publish_event(job, event.model_dump(mode="json"))
            self._logger.info("[JOB_SUCCEEDED]", f"request_id={request_id}", f"duration_ms={duration_ms}")
            return True
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            error_message = str(exc)
            self._job_repository.make_job_failed(request_id, error_message)
            event = self._build_failed_event(job, error_message)
            self._publish_event(job, event.model_dump(mode="json"))
            self._logger.error(
                "[JOB_FAILED]",
                f"request_id={request_id}",
                f"error={error_message}",
                f"duration_ms={duration_ms}",
            )
            return True

    def _publish_event(self, job: dict[str, Any], value: dict[str, Any]) -> None:
        self._producer.publish(
            topic=self._event_topic,
            key=self._get_publish_key(job),
            value=value,
        )

    @staticmethod
    def _get_request_id(job: dict[str, Any]) -> str:
        return str(job["request_id"])

    @staticmethod
    def _build_running_log_fields(job: dict[str, Any]) -> list[str]:
        return [
            f"{key}={value}"
            for key, value in job.items()
            if key in {"request_id", "model_name", "dataset_version", "model_version"} and value is not None
        ]

    @abstractmethod
    def _get_next_job(self) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _build_completed_event(self, job: dict[str, Any], result: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _build_failed_event(self, job: dict[str, Any], error_message: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_publish_key(self, job: dict[str, Any]) -> str:
        raise NotImplementedError
