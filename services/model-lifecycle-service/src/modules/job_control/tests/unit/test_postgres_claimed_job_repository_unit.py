from __future__ import annotations

import sys
import unittest
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.platform.logger import Logger  # noqa: E402
from src.modules.job_control.adapters.outbound.persistence.postgres.postgres_claimed_job_repository import (  # noqa: E402
    DEFAULT_JOB_RETURNING,
    PostgresClaimedJobRepository,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.fetchrow_result = None
        self.fetchrow_calls: list[tuple[object, tuple[object, ...]]] = []

    async def fetchrow(self, query: object, *args: object, **params: object) -> object:
        self.fetchrow_calls.append((query, tuple(args)))
        return self.fetchrow_result


class _FakeTransaction:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def transaction(self):
        yield self._connection


class PostgresClaimedJobRepositoryUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_claim_next_pending_job_uses_configured_event_type(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {"request_id": "req-1", "status": "PROCESSING"}
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="yolo_detector",
        )

        claimed = await repository.claim_next_pending_job(server_id="worker-1")

        assert claimed is not None
        self.assertEqual(claimed.request_id, "req-1")
        self.assertEqual(claimed.event_type, "yolo_detector")
        self.assertEqual(claimed.status, "PROCESSING")
        query, params = connection.fetchrow_calls[0]
        self.assertIn("e.event_type = 'yolo_detector'", query)
        self.assertIn("e.payload->>'git_commit' AS git_commit", query)
        self.assertIsInstance(params[0], datetime)
        self.assertEqual(params[1], "worker-1")
        self.assertIsInstance(params[2], datetime)
        self.assertEqual(len(params), 3)

    async def test_claim_next_pending_job_binds_stale_cutoff_parameter(self) -> None:
        connection = _FakeConnection()
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="yolo_detector",
            reclaim_timeout_seconds=1800,
        )

        await repository.claim_next_pending_job(server_id="worker-1")

        query, params = connection.fetchrow_calls[0]
        self.assertIn("e.updated_at < $1", query)
        self.assertIn("running.updated_at >= $3", query)
        self.assertEqual(params[1], "worker-1")
        self.assertIsNotNone(params[0])
        self.assertIsNotNone(params[2])

    async def test_claim_next_pending_job_adds_legacy_payload_event_filter_when_requested(self) -> None:
        connection = _FakeConnection()
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="vit_ctc_deepseek",
            event_filter="tracking",
        )

        await repository.claim_next_pending_job(server_id="worker-1")

        query, _ = connection.fetchrow_calls[0]
        self.assertIn("AND e.payload->>'event' = 'tracking'", query)

    async def test_claim_next_pending_job_adds_model_name_filter(self) -> None:
        connection = _FakeConnection()
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="yolo_detector",
            model_name_filter="detector",
        )

        await repository.claim_next_pending_job(server_id="worker-1")

        query, _ = connection.fetchrow_calls[0]
        self.assertIn("AND e.payload->>'model_name' = 'detector'", query)

    def test_default_job_returning_contains_expected_projection(self) -> None:
        self.assertIn("e.payload->>'model_name' AS model_name", DEFAULT_JOB_RETURNING)
        self.assertIn("e.payload->>'checkpoint_best_name' AS checkpoint_best_name", DEFAULT_JOB_RETURNING)
        self.assertIn("e.payload->>'output_path' AS output_path", DEFAULT_JOB_RETURNING)
        self.assertIn("e.payload->>'event' AS event", DEFAULT_JOB_RETURNING)

    async def test_claim_next_pending_job_includes_download_output_fields_in_payload(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {
            "request_id": "req-download-1",
            "status": "PROCESSING",
            "model_name": "vit_ctc_deepseek",
            "model_version": "",
            "git_commit": "abc123",
            "checkpoint_best_name": "best_cer.pt",
            "checkpoint_last_name": "last_checkpoint.pt",
            "output_path": "data/test-downloads/vit_ctc_deepseek/best_cer.pt",
            "event": "download_requested",
            "version": "1.0",
        }
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="vit_ctc_deepseek",
            event_filter="download_requested",
        )

        claimed = await repository.claim_next_pending_job(server_id="worker-1")

        assert claimed is not None
        self.assertEqual(
            claimed.payload["output_path"],
            "data/test-downloads/vit_ctc_deepseek/best_cer.pt",
        )

    async def test_claim_next_pending_job_accepts_string_returning_projection(self) -> None:
        connection = _FakeConnection()
        connection.fetchrow_result = {
            "request_id": "req-cl-1",
            "status": "PROCESSING",
            "raw_dir": "/tmp/raw",
            "output_dir": "/tmp/out",
            "class_id": "0",
        }
        repository = PostgresClaimedJobRepository(
            transaction=_FakeTransaction(connection),
            logger=Logger(name="test.claimed_job_repo"),
            event_type="continual_learning_requested",
            returning="""
e.id,
e.request_id,
e.payload->>'raw_dir' AS raw_dir,
e.payload->>'output_dir' AS output_dir,
e.payload->>'class_id' AS class_id,
e.status
""",
            payload_builder=lambda row: {
                "raw_dir": row.get("raw_dir"),
                "output_dir": row.get("output_dir"),
                "class_id": row.get("class_id"),
            },
        )

        claimed = await repository.claim_next_pending_job(server_id="worker-1")

        assert claimed is not None
        self.assertEqual(claimed.request_id, "req-cl-1")
        self.assertEqual(claimed.payload["raw_dir"], "/tmp/raw")
        query, _ = connection.fetchrow_calls[0]
        self.assertIn("e.request_id", query)
        self.assertIn("e.payload->>'raw_dir' AS raw_dir", query)


if __name__ == "__main__":
    unittest.main()
