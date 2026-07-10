from __future__ import annotations

from src.modules.job_control.application.dto.consume_event_command import (
    ConsumeEventCommand,
)
from src.modules.job_control.application.ports.job_record_repository import (
    JobRecordRepository,
)
from src.modules.job_control.domain.entity_objects.job_record import JobRecord
from src.modules.job_control.domain.value_objects.job_status import JobStatus
from src.modules.job_control.domain.value_objects.local_time import local_now
from src.platform.logger import Logger


class IngestJob:
    def __init__(
        self,
        *,
        repository: JobRecordRepository,
        logger: Logger,
        expected_event_types: list[str],
    ) -> None:
        self._repository = repository
        self._logger = logger
        self._expected_event_types = expected_event_types

    async def execute(self, command: ConsumeEventCommand) -> bool:
        if command.event_type not in self._expected_event_types:
            raise ValueError(
                f"Unexpected event_type={command.event_type}; "
                f"expected one of {self._expected_event_types}"
            )

        job = JobRecord(
            request_id=command.request_id,
            message_identity=command.message_identity,
            event_type=command.event_type,
            payload=command.payload,
            status=JobStatus.RECEIVED,
            produced_at=local_now(),
        )
        created = await self._repository.create_job_if_not_exists(
            job=job,
            schema_name=command.schema_name,
            schema_version=command.schema_version,
            message_key=command.message_key,
        )
        self._logger.info(
            f"[{type(self).__name__}]",
            f"request_id={command.request_id}",
            f"event_type={command.event_type}",
            f"created={created}",
        )
        return created
