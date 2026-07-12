from __future__ import annotations

import json
from typing import Any

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.domain.value_objects.local_time import local_now
from src.platform.logger import Logger
from src.platform.messaging.kafka.protocols import MessageProducer


class JobSuccessEventPublisher:
    def __init__(
        self,
        *,
        producer: MessageProducer,
        topic: str,
        logger: Logger,
    ) -> None:
        self._producer = producer
        self._topic = topic
        self._logger = logger

    def publish(
        self,
        *,
        job_name: str,
        server_id: str,
        claimed_job: ClaimedJobDto,
        result: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "request_id": claimed_job.request_id,
            "job_type": job_name,
            "event_type": claimed_job.event_type,
            "status": "PROCESSED",
            "server_id": server_id,
            "processed_at": local_now().isoformat(),
            "payload": claimed_job.payload,
            "result": result or {"success": True},
            "error_message": None,
        }
        self._producer.publish(
            topic=self._topic,
            key=claimed_job.request_id.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
            headers=[("content-type", b"application/json")],
        )
        self._logger.info(
            "[JOB_SUCCESS_EVENT_PUBLISHED]",
            f"topic={self._topic}",
            f"request_id={claimed_job.request_id}",
            f"event_type={claimed_job.event_type}",
            f"job_type={job_name}",
        )
