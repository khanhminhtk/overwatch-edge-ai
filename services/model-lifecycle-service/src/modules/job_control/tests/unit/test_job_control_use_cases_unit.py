from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.job_control.application.dto.consume_event_command import (  # noqa: E402
    ConsumeEventCommand,
)
from src.modules.job_control.application.dto.job_result_dto import JobResultDto  # noqa: E402
from src.modules.job_control.application.use_case.claim_mlflow_tracking_job import (  # noqa: E402
    ClaimMlflowTrackingJob,
)
from src.modules.job_control.application.use_case.ingest_job import (  # noqa: E402
    IngestJob,
)
from src.modules.job_control.application.use_case.mark_job_failed import (  # noqa: E402
    MarkJobFailed,
)
from src.modules.job_control.application.use_case.mark_job_processed import (  # noqa: E402
    MarkJobProcessed,
)
from src.modules.job_control.application.use_case.process_next_job import (  # noqa: E402
    ProcessNextJob,
)
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity  # noqa: E402


class JobControlUseCasesUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_mlflow_tracking_job_creates_job_record(self) -> None:
        repository = MagicMock()
        repository.create_job_if_not_exists = AsyncMock(return_value=True)
        logger = MagicMock()
        use_case = IngestJob(
            repository=repository,
            logger=logger,
            expected_event_types=["yolo_detector"],
        )

        created = await use_case.execute(
            ConsumeEventCommand(
                request_id="req-1",
                event_type="yolo_detector",
                schema_name="mlflow_tracking_job",
                schema_version="1.0",
                message_key="yolo_detector",
                message_identity=MessageIdentity(
                    topic="topic-a",
                    partition_id=1,
                    message_offset=9,
                    consumer_group="group-a",
                ),
                payload={"model_name": "yolo_detector"},
            )
        )

        self.assertTrue(created)
        call = repository.create_job_if_not_exists.await_args
        self.assertEqual(call.kwargs["job"].request_id, "req-1")
        self.assertEqual(call.kwargs["job"].event_type, "yolo_detector")
        self.assertEqual(call.kwargs["schema_name"], "mlflow_tracking_job")

    async def test_claim_mlflow_tracking_job_delegates_to_repository(self) -> None:
        repository = MagicMock()
        repository.claim_next_pending_job = AsyncMock(
            return_value=ClaimedJobDto(
                request_id="req-2",
                event_type="vit_ctc_deepseek",
                payload={"model_name": "vit_ctc_deepseek"},
                status="PROCESSING",
            )
        )
        use_case = ClaimMlflowTrackingJob(repository=repository)

        claimed = await use_case.execute(server_id="worker-1")

        self.assertEqual(claimed.request_id, "req-2")
        repository.claim_next_pending_job.assert_awaited_once_with(server_id="worker-1")

    async def test_mark_job_processed_updates_repository(self) -> None:
        repository = MagicMock()
        repository.mark_processed_by_request_id = AsyncMock(return_value=True)
        use_case = MarkJobProcessed(repository=repository)

        updated = await use_case.execute(request_id="req-3")

        self.assertTrue(updated)
        repository.mark_processed_by_request_id.assert_awaited_once_with("req-3")

    async def test_mark_job_failed_updates_repository(self) -> None:
        repository = MagicMock()
        repository.mark_failed_by_request_id = AsyncMock(return_value=True)
        use_case = MarkJobFailed(repository=repository)

        updated = await use_case.execute(request_id="req-4", error_message="boom")

        self.assertTrue(updated)
        repository.mark_failed_by_request_id.assert_awaited_once_with(
            request_id="req-4",
            error_message="boom",
        )

    async def test_process_next_job_marks_processed_on_success(self) -> None:
        claim_use_case = MagicMock()
        claim_use_case.execute = AsyncMock(
            return_value=ClaimedJobDto(
                request_id="req-5",
                event_type="yolo_detector",
                payload={"model_name": "yolo_detector"},
                status="PROCESSING",
            )
        )
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=JobResultDto(request_id="req-5", success=True))
        mark_processed = MagicMock()
        mark_processed.execute = AsyncMock(return_value=True)
        mark_failed = MagicMock()
        mark_failed.execute = AsyncMock(return_value=False)
        logger = MagicMock()
        use_case = ProcessNextJob(
            claim_job=claim_use_case,
            handler=handler,
            mark_processed=mark_processed,
            mark_failed=mark_failed,
            logger=logger,
        )

        with patch(
            "src.modules.job_control.application.use_case.process_next_job.time.perf_counter",
            side_effect=[10.0, 12.5],
        ):
            result = await use_case.execute(server_id="worker-1")

        self.assertEqual(result.request_id, "req-5")
        mark_processed.execute.assert_awaited_once_with(request_id="req-5")
        mark_failed.execute.assert_not_awaited()
        logger.info.assert_any_call(
            "[PROCESS_NEXT_JOB_HANDLER_COMPLETED]",
            "server_id=worker-1",
            "request_id=req-5",
            "success=True",
            "duration_ms=2500.00",
        )
        logger.info.assert_any_call(
            "[PROCESS_NEXT_JOB_STATE_PERSISTED]",
            "server_id=worker-1",
            "request_id=req-5",
            "target_status=PROCESSED",
            "attempt=1",
        )

    async def test_process_next_job_marks_failed_on_unsuccessful_result(self) -> None:
        claim_use_case = MagicMock()
        claim_use_case.execute = AsyncMock(
            return_value=ClaimedJobDto(
                request_id="req-6",
                event_type="yolo_detector",
                payload={"model_name": "yolo_detector"},
                status="PROCESSING",
            )
        )
        handler = MagicMock()
        handler.handle = AsyncMock(
            return_value=JobResultDto(
                request_id="req-6",
                success=False,
                error_message="tracking failed",
            )
        )
        mark_processed = MagicMock()
        mark_processed.execute = AsyncMock(return_value=False)
        mark_failed = MagicMock()
        mark_failed.execute = AsyncMock(return_value=True)
        logger = MagicMock()
        use_case = ProcessNextJob(
            claim_job=claim_use_case,
            handler=handler,
            mark_processed=mark_processed,
            mark_failed=mark_failed,
            logger=logger,
        )

        with patch(
            "src.modules.job_control.application.use_case.process_next_job.time.perf_counter",
            side_effect=[3.0, 3.125],
        ):
            result = await use_case.execute(server_id="worker-1")

        self.assertEqual(result.request_id, "req-6")
        mark_failed.execute.assert_awaited_once_with(
            request_id="req-6",
            error_message="tracking failed",
        )
        mark_processed.execute.assert_not_awaited()
        logger.info.assert_any_call(
            "[PROCESS_NEXT_JOB_HANDLER_COMPLETED]",
            "server_id=worker-1",
            "request_id=req-6",
            "success=False",
            "duration_ms=125.00",
        )
        logger.info.assert_any_call(
            "[PROCESS_NEXT_JOB_STATE_PERSISTED]",
            "server_id=worker-1",
            "request_id=req-6",
            "target_status=FAILED",
            "attempt=1",
        )

    async def test_process_next_job_returns_none_when_no_job_claimed(self) -> None:
        claim_use_case = MagicMock()
        claim_use_case.execute = AsyncMock(return_value=None)
        handler = MagicMock()
        mark_processed = MagicMock()
        mark_failed = MagicMock()
        logger = MagicMock()
        use_case = ProcessNextJob(
            claim_job=claim_use_case,
            handler=handler,
            mark_processed=mark_processed,
            mark_failed=mark_failed,
            logger=logger,
        )

        result = await use_case.execute(server_id="worker-1")

        self.assertIsNone(result)
        handler.handle.assert_not_called()

    async def test_process_next_job_retries_mark_processed_after_timeout(self) -> None:
        claim_use_case = MagicMock()
        claim_use_case.execute = AsyncMock(
            return_value=ClaimedJobDto(
                request_id="req-7",
                event_type="yolo_detector",
                payload={"model_name": "yolo_detector"},
                status="PROCESSING",
            )
        )
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=JobResultDto(request_id="req-7", success=True))
        mark_processed = MagicMock()
        mark_processed.execute = AsyncMock(side_effect=[TimeoutError("db locked"), True])
        mark_failed = MagicMock()
        mark_failed.execute = AsyncMock(return_value=False)
        logger = MagicMock()
        use_case = ProcessNextJob(
            claim_job=claim_use_case,
            handler=handler,
            mark_processed=mark_processed,
            mark_failed=mark_failed,
            logger=logger,
        )

        with patch(
            "src.modules.job_control.application.use_case.process_next_job.time.perf_counter",
            side_effect=[10.0, 10.2],
        ), patch(
            "src.modules.job_control.application.use_case.process_next_job.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            result = await use_case.execute(server_id="worker-1")

        self.assertTrue(result.success)
        self.assertEqual(mark_processed.execute.await_count, 2)
        sleep_mock.assert_awaited_once_with(1.0)
        logger.warning.assert_any_call(
            "[PROCESS_NEXT_JOB_STATE_PERSIST_RETRY]",
            "server_id=worker-1",
            "request_id=req-7",
            "target_status=PROCESSED",
            "attempt=1",
            "error=db locked",
        )

    async def test_process_next_job_raises_after_mark_processed_retries_exhausted(self) -> None:
        claim_use_case = MagicMock()
        claim_use_case.execute = AsyncMock(
            return_value=ClaimedJobDto(
                request_id="req-8",
                event_type="yolo_detector",
                payload={"model_name": "yolo_detector"},
                status="PROCESSING",
            )
        )
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=JobResultDto(request_id="req-8", success=True))
        mark_processed = MagicMock()
        mark_processed.execute = AsyncMock(side_effect=TimeoutError("db locked"))
        mark_failed = MagicMock()
        mark_failed.execute = AsyncMock(return_value=False)
        logger = MagicMock()
        use_case = ProcessNextJob(
            claim_job=claim_use_case,
            handler=handler,
            mark_processed=mark_processed,
            mark_failed=mark_failed,
            logger=logger,
        )

        with patch(
            "src.modules.job_control.application.use_case.process_next_job.time.perf_counter",
            side_effect=[2.0, 2.1],
        ), patch(
            "src.modules.job_control.application.use_case.process_next_job.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            with self.assertRaisesRegex(TimeoutError, "db locked"):
                await use_case.execute(server_id="worker-1")

        self.assertEqual(mark_processed.execute.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)
        logger.exception.assert_any_call(
            "[PROCESS_NEXT_JOB_STATE_PERSIST_FAILED]",
            "server_id=worker-1",
            "request_id=req-8",
            "target_status=PROCESSED",
            "attempts=3",
        )


if __name__ == "__main__":
    unittest.main()
