from __future__ import annotations

from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_claim_sql import (
    CLAIM_NEXT_PENDING_JOB,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_insert_sql import (
    INSERT_JOB_EVENT,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_select_sql import (
    SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY,
    SELECT_JOB_EVENTS_BY_EVENT_TYPE,
    SELECT_JOB_EVENTS_BY_REQUEST_ID,
    SELECT_JOB_EVENTS_BY_STATUS,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_update_sql import (
    MARK_JOB_EVENT_FAILED,
    MARK_JOB_EVENT_PROCESSED,
    MARK_JOB_EVENT_PROCESSING,
    UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID,
    UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID,
    UPDATE_JOB_EVENT_STATUS,
)

__all__ = [
    "CLAIM_NEXT_PENDING_JOB",
    "INSERT_JOB_EVENT",
    "MARK_JOB_EVENT_FAILED",
    "MARK_JOB_EVENT_PROCESSING",
    "MARK_JOB_EVENT_PROCESSED",
    "SELECT_JOB_EVENTS_BY_REQUEST_ID",
    "SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY",
    "SELECT_JOB_EVENTS_BY_STATUS",
    "SELECT_JOB_EVENTS_BY_EVENT_TYPE",
    "UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID",
    "UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID",
    "UPDATE_JOB_EVENT_STATUS",
]
