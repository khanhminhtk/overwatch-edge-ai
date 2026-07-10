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
    MlflowTrackingProcessor,
    RecognizerTrackingJobProcessor,
)


class MlflowTrackingProcessorMainUnitTest(unittest.TestCase):
    def test_recognizer_tracking_is_alias_for_mlflow_tracking(self) -> None:
        self.assertIs(RecognizerTrackingJobProcessor, MlflowTrackingProcessor)

    def test_mlflow_tracking_processor_api(self) -> None:
        runner = MagicMock()
        runner.run_once = AsyncMock(return_value=JobResultDto(request_id="req-1", success=True))
        runner.run_forever = AsyncMock(return_value=None)
        runner.stop = MagicMock()

        processor = MlflowTrackingProcessor(runner=runner)

        self.assertTrue(hasattr(processor, "run_once"))
        self.assertTrue(hasattr(processor, "run_forever"))
        self.assertTrue(hasattr(processor, "stop"))


if __name__ == "__main__":
    unittest.main()
