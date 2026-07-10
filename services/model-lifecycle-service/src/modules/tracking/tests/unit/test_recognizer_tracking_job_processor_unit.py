from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.job_result_dto import JobResultDto  # noqa: E402
from src.modules.tracking.adapters.inbound.job_control.tracking_job_processor import (  # noqa: E402
    RecognizerTrackingJobProcessor,
)


class RecognizerTrackingJobProcessorUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_delegates_to_runner(self) -> None:
        runner = MagicMock()
        runner.run_once = AsyncMock(return_value=JobResultDto(request_id="req-1", success=True))
        runner.run_forever = AsyncMock(return_value=None)
        runner.stop = MagicMock()
        processor = RecognizerTrackingJobProcessor(runner=runner)

        result = await processor.run_once()

        self.assertEqual(result.request_id, "req-1")
        runner.run_once.assert_awaited_once_with()

    async def test_run_forever_delegates_to_runner(self) -> None:
        runner = MagicMock()
        runner.run_once = AsyncMock(return_value=None)
        runner.run_forever = AsyncMock(return_value=None)
        runner.stop = MagicMock()
        processor = RecognizerTrackingJobProcessor(runner=runner)

        await processor.run_forever()

        runner.run_forever.assert_awaited_once_with()

    def test_stop_delegates_to_runner(self) -> None:
        runner = MagicMock()
        runner.run_once = AsyncMock(return_value=None)
        runner.run_forever = AsyncMock(return_value=None)
        runner.stop = MagicMock()
        processor = RecognizerTrackingJobProcessor(runner=runner)

        processor.stop()

        runner.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
