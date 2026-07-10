from __future__ import annotations

import sys
import unittest
from pathlib import Path

from psycopg import sql


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_claim_sql import (  # noqa: E402
    CLAIM_NEXT_PENDING_JOB,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_insert_sql import (  # noqa: E402
    INSERT_JOB_EVENT,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_select_sql import (  # noqa: E402
    SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY,
    SELECT_JOB_EVENTS_BY_REQUEST_ID,
    SELECT_JOB_EVENTS_BY_STATUS,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.queries.job_event_update_sql import (  # noqa: E402
    MARK_JOB_EVENT_FAILED,
    MARK_JOB_EVENT_PROCESSED,
    MARK_JOB_EVENT_PROCESSING,
    UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID,
    UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID,
)


class JobControlQueriesUnitTest(unittest.TestCase):
    def test_insert_job_event_has_deduplication_clause(self) -> None:
        self.assertIn("INSERT INTO kafka_events", INSERT_JOB_EVENT)
        self.assertIn("ON CONFLICT", INSERT_JOB_EVENT)
        self.assertIn("DO NOTHING", INSERT_JOB_EVENT)
        self.assertNotIn("ON CONFLICT (", INSERT_JOB_EVENT)

    def test_select_job_event_queries_cover_request_and_message_identity(self) -> None:
        self.assertIn("WHERE request_id = %(request_id)s", SELECT_JOB_EVENTS_BY_REQUEST_ID)
        self.assertIn("WHERE topic = %(topic)s", SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY)
        self.assertIn("message_offset = %(message_offset)s", SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY)
        self.assertIn("status = %(status)s", SELECT_JOB_EVENTS_BY_STATUS)

    def test_update_job_event_queries_cover_processing_success_and_failure(self) -> None:
        self.assertIn("status = 'PROCESSING'", MARK_JOB_EVENT_PROCESSING)
        self.assertIn("status = 'PROCESSED'", MARK_JOB_EVENT_PROCESSED)
        self.assertIn("status = 'FAILED'", MARK_JOB_EVENT_FAILED)
        self.assertIn("Asia/Ho_Chi_Minh", MARK_JOB_EVENT_PROCESSING)
        self.assertIn("Asia/Ho_Chi_Minh", MARK_JOB_EVENT_PROCESSED)
        self.assertIn("Asia/Ho_Chi_Minh", MARK_JOB_EVENT_FAILED)
        self.assertIn("WHERE request_id = %(request_id)s", UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID)
        self.assertIn("error_message = %(error_message)s", UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID)

    def test_claim_next_pending_job_uses_skip_locked_pattern(self) -> None:
        self.assertIsInstance(CLAIM_NEXT_PENDING_JOB, sql.SQL)

        query_text = str(CLAIM_NEXT_PENDING_JOB)
        self.assertIn("FOR UPDATE SKIP LOCKED", query_text)
        self.assertIn("e.status = 'RECEIVED'", query_text)
        self.assertIn("status = 'PROCESSING'", query_text)
        self.assertIn("running.updated_at >= %(stale_cutoff)s", query_text)
        self.assertIn("e.updated_at < %(stale_cutoff)s", query_text)
        self.assertIn("Asia/Ho_Chi_Minh", query_text)
        self.assertIn("{event_type}", query_text)
        self.assertIn("{additional_conditions}", query_text)
        self.assertIn("{returning}", query_text)


if __name__ == "__main__":
    unittest.main()
