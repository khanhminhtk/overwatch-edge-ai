from __future__ import annotations

import sys
import unittest
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from psycopg import sql


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_claim_repository import (  # noqa: E402
    PostgresJobClaimRepository,
)
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_job_record_repository import (  # noqa: E402
    PostgresJobRecordRepository,
)
from src.modules.job_control.domain.entity_objects.job_record import JobRecord  # noqa: E402
from src.modules.job_control.domain.value_objects.job_status import JobStatus  # noqa: E402
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity  # noqa: E402
from src.platform.logger import Logger  # noqa: E402


class _FakeConnection:
    def __init__(self) -> None:
        self.fetchrow_result = None
        self.fetch_result = []
        self.fetchrow_calls: list[tuple[object, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[object, tuple[object, ...]]] = []

    async def fetchrow(self, query: object, *args: object, **params: object) -> object:
        self.fetchrow_calls.append((query, tuple(args)))
        return self.fetchrow_result

    async def fetch(self, query: object, *args: object, **params: object) -> list[object]:
        self.fetch_calls.append((query, tuple(args)))
        return list(self.fetch_result)


class _FakeTransaction:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def transaction(self):
        yield self._connection


class PostgresJobRepositoriesUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_record_repository_inserts_job_and_returns_created_flag(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {"id": 101}
        repository = PostgresJobRecordRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.record_repo"),
        )
        job = JobRecord(
            request_id="req-1",
            message_identity=MessageIdentity(
                topic="model.lifecycle.requests",
                partition_id=1,
                message_offset=22,
                consumer_group="trainer",
            ),
            event_type="train_requested",
            payload={"model_name": "detector"},
            status=JobStatus.RECEIVED,
        )

        created = await repository.create_job_if_not_exists(
            job=job,
            schema_name="training_event",
            schema_version="1.0",
            message_key="detector",
        )

        self.assertTrue(created)
        query, args = connection.fetchrow_calls[0]
        self.assertIn("INSERT INTO kafka_events", query)
        self.assertIn("$1", query)
        self.assertEqual(args[0], "req-1")
        self.assertEqual(args[1], "model.lifecycle.requests")
        self.assertEqual(args[4], "trainer")
        self.assertEqual(args[10], JobStatus.RECEIVED.value)

    async def test_record_repository_loads_rows_by_request_id(self) -> None:
        connection = _FakeConnection()
        connection.fetch_result = [
            {"request_id": "req-1", "status": "RECEIVED"},
            {"request_id": "req-1", "status": "FAILED"},
        ]
        repository = PostgresJobRecordRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.record_repo"),
        )

        rows = await repository.get_by_request_id("req-1")

        self.assertEqual(len(rows), 2)
        query, args = connection.fetch_calls[0]
        self.assertIn("WHERE request_id = $1", query)
        self.assertEqual(args, ("req-1",))

    async def test_record_repository_marks_failed_by_request_id(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {"id": 101}
        repository = PostgresJobRecordRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.record_repo"),
        )

        updated = await repository.mark_failed_by_request_id(
            request_id="req-1",
            error_message="boom",
        )

        self.assertTrue(updated)
        query, args = connection.fetchrow_calls[0]
        self.assertIn("status = 'FAILED'", query)
        self.assertEqual(args, ("boom", "req-1"))

    async def test_claim_repository_claims_next_pending_job(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {"id": 7, "request_id": "req-2", "status": "PROCESSING"}
        logger = MagicMock()
        repository = PostgresJobClaimRepository(
            transaction=_FakeTransaction(connection),
            logger=logger,
        )

        claimed = await repository.claim_next_pending_job(
            server_id="worker-1",
            event_type="train_requested",
            reclaim_timeout_seconds=1800,
            returning=sql.SQL("e.id, e.request_id, e.status"),
        )

        self.assertEqual(claimed, {"id": 7, "request_id": "req-2", "status": "PROCESSING"})
        query, args = connection.fetchrow_calls[0]
        self.assertIn("FOR UPDATE SKIP LOCKED", query)
        self.assertIn("e.event_type = 'train_requested'", query)
        self.assertIsInstance(args[0], datetime)
        self.assertEqual(args[1], "worker-1")
        self.assertIsInstance(args[2], datetime)
        logger.info.assert_any_call(
            "[JOB_EVENT_CLAIM_ATTEMPT]",
            "event_type=train_requested",
            "server_id=worker-1",
            f"stale_cutoff={args[0].isoformat()}",
            "claimed=True",
            "request_id=req-2",
            "status=PROCESSING",
        )

    async def test_claim_repository_logs_claim_miss(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = None
        logger = MagicMock()
        repository = PostgresJobClaimRepository(
            transaction=_FakeTransaction(connection),
            logger=logger,
        )

        claimed = await repository.claim_next_pending_job(
            server_id="worker-1",
            event_type="train_requested",
            reclaim_timeout_seconds=1800,
            returning=sql.SQL("e.id, e.request_id, e.status"),
        )

        self.assertIsNone(claimed)
        logger.info.assert_any_call(
            "[JOB_EVENT_CLAIM_ATTEMPT]",
            "event_type=train_requested",
            "server_id=worker-1",
            f"stale_cutoff={connection.fetchrow_calls[0][1][0].isoformat()}",
            "claimed=False",
        )


if __name__ == "__main__":
    unittest.main()
