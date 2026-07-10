from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.adapters.outbound.scheduler.polling_job_runner import (  # noqa: E402
    PollingJobRunner,
)
from src.modules.job_control.application.dto.job_result_dto import JobResultDto  # noqa: E402


class PollingJobRunnerUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_delegates_to_process_next_job(self) -> None:
        process_next_job = MagicMock()
        process_next_job.execute = AsyncMock(
            return_value=JobResultDto(request_id="req-1", success=True)
        )
        logger = MagicMock()
        runner = PollingJobRunner(
            process_next_job=process_next_job,
            server_id="worker-1",
            event_type="yolo_detector",
            logger=logger,
            idle_sleep_seconds=0.1,
        )

        result = await runner.run_once()

        self.assertEqual(result.request_id, "req-1")
        process_next_job.execute.assert_awaited_once_with(server_id="worker-1")
        info_messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertIn("[JOB_POLLING_TICK]", info_messages)
        self.assertIn("[JOB_POLLING_RESULT]", info_messages)
        tick_call = logger.info.call_args_list[0]
        self.assertEqual(tick_call.args[1], "server_id=worker-1")
        self.assertEqual(tick_call.args[2], "event_type=yolo_detector")

    async def test_run_forever_sleeps_when_no_job(self) -> None:
        process_next_job = MagicMock()
        process_next_job.execute = AsyncMock(side_effect=[None, None])
        logger = MagicMock()
        runner = PollingJobRunner(
            process_next_job=process_next_job,
            server_id="worker-1",
            event_type="yolo_detector",
            logger=logger,
            idle_sleep_seconds=0.1,
        )

        async def _sleep(_: float) -> None:
            runner.stop()

        with patch(
            "src.modules.job_control.adapters.outbound.scheduler.polling_job_runner.asyncio.sleep",
            new=AsyncMock(side_effect=_sleep),
        ) as sleep_mock:
            await runner.run_forever()

        sleep_mock.assert_awaited()
        idle_calls = [
            call for call in logger.info.call_args_list if call.args[0] == "[JOB_POLLING_IDLE]"
        ]
        self.assertTrue(idle_calls)
        self.assertEqual(idle_calls[0].args[2], "event_type=yolo_detector")

    async def test_stop_sets_flag(self) -> None:
        process_next_job = MagicMock()
        process_next_job.execute = AsyncMock(return_value=None)
        logger = MagicMock()
        runner = PollingJobRunner(
            process_next_job=process_next_job,
            server_id="worker-1",
            event_type="yolo_detector",
            logger=logger,
            idle_sleep_seconds=0.1,
        )

        runner.stop()

        self.assertTrue(runner._stop_requested)
        logger.info.assert_called_once()

    async def test_run_forever_logs_and_recovers_from_iteration_error(self) -> None:
        process_next_job = MagicMock()
        process_next_job.execute = AsyncMock(side_effect=[TimeoutError("db locked"), None])
        logger = MagicMock()
        runner = PollingJobRunner(
            process_next_job=process_next_job,
            server_id="worker-1",
            event_type="yolo_detector",
            logger=logger,
            idle_sleep_seconds=0.1,
        )

        async def _sleep(_: float) -> None:
            if process_next_job.execute.await_count >= 2:
                runner.stop()

        with patch(
            "src.modules.job_control.adapters.outbound.scheduler.polling_job_runner.asyncio.sleep",
            new=AsyncMock(side_effect=_sleep),
        ) as sleep_mock:
            await runner.run_forever()

        self.assertGreaterEqual(sleep_mock.await_count, 2)
        logger.exception.assert_any_call(
            "[JOB_POLLING_ERROR]",
            "server_id=worker-1",
            "event_type=yolo_detector",
        )


if __name__ == "__main__":
    unittest.main()
