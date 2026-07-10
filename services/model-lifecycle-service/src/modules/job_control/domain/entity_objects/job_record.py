
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.modules.job_control.domain.errors.invalid_job_state_error import (
    InvalidJobStateError,
)
from src.modules.job_control.domain.value_objects.job_status import JobStatus
from src.modules.job_control.domain.value_objects.local_time import local_now
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity


@dataclass
class JobRecord:
    request_id: str
    message_identity: MessageIdentity
    event_type: str
    payload: dict | None
    status: JobStatus = JobStatus.RECEIVED
    error_message: str | None = None
    produced_at: datetime | None = None
    processed_at: datetime | None = None
    claimed_by: str | None = None

    def mark_processing(self, server_id: str) -> None:
        if self.status is not JobStatus.RECEIVED:
            raise InvalidJobStateError(
                f"cannot mark processing from status={self.status}"
            )
        self.status = JobStatus.PROCESSING
        self.claimed_by = server_id
        self.error_message = None

    def mark_processed(self) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise InvalidJobStateError(
                f"cannot mark processed from status={self.status}"
            )
        self.status = JobStatus.PROCESSED
        self.processed_at = local_now()

    def mark_failed(self, error_message: str) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise InvalidJobStateError(
                f"cannot mark failed from status={self.status}"
        )
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.processed_at = local_now()
