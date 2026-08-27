from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.exports.adapters.inbound.job_control.export_job_handler import (  # noqa: E402
    ExportJobHandler,
)
from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.tracking.domain.value_objects import DetectionConfig, RecognizerConfig  # noqa: E402
from src.platform.logger import Logger  # noqa: E402


class ExportJobHandlerUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.detection_export = MagicMock()
        self.recognizer_export = MagicMock()
        self.logger = MagicMock(spec=Logger)
        self.handler = ExportJobHandler(
            detection_export=self.detection_export,
            recognizer_export=self.recognizer_export,
            detection_config=DetectionConfig(
                checkpoint_dir="artifacts/detection",
                best_checkpoint_name="best.pt",
                training_config_path="ml/training/config/detect.yaml",
                training_env_path="ml/training/.env",
            ),
            recognizer_config=RecognizerConfig(
                checkpoint_dir="artifacts/recognizer",
                best_checkpoint_name="best_cer.pt",
                training_config_path="ml/training/config/recog.yaml",
                training_env_path="ml/training/.env",
            ),
            detection_event_type="yolo_detector",
            recognizer_event_type="vit_ctc_deepseek",
            project_root=Path("/tmp/export-job-handler-test"),
            logger=self.logger,
        )

    async def test_handle_detection_job_builds_expected_export_spec(self) -> None:
        with (
            patch(
                "src.modules.exports.adapters.inbound.job_control.export_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda func, *args: func(*args)),
            ),
            patch.object(Path, "exists", autospec=True, return_value=True),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-1",
                    event_type="yolo_detector",
                    payload={"checkpoint_best_name": "best-custom.pt"},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        self.detection_export.execute.assert_called_once()
        spec = self.detection_export.execute.call_args.args[0]
        self.assertEqual(spec.model_type, "detection")
        self.assertEqual(spec.checkpoint_path, Path("/tmp/export-job-handler-test/artifacts/detection/best-custom.pt"))
        self.assertEqual(spec.output_path, Path("/tmp/export-job-handler-test/artifacts/onnx/detection.onnx"))
        self.assertEqual(spec.training_config_path, Path("/tmp/export-job-handler-test/ml/training/config/detect.yaml"))
        self.assertEqual(spec.training_env_path, Path("/tmp/export-job-handler-test/ml/training/.env"))
        self.recognizer_export.execute.assert_not_called()

    async def test_handle_recognizer_job_uses_default_checkpoint_name_when_missing(self) -> None:
        with patch(
            "src.modules.exports.adapters.inbound.job_control.export_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-2",
                    event_type="vit_ctc_deepseek",
                    payload={},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        self.recognizer_export.execute.assert_called_once()
        spec = self.recognizer_export.execute.call_args.args[0]
        self.assertEqual(spec.model_type, "recognizer")
        self.assertEqual(spec.checkpoint_path, Path("/tmp/export-job-handler-test/artifacts/recognizer/best_cer.pt"))

    async def test_handle_falls_back_to_default_checkpoint_when_payload_checkpoint_missing(self) -> None:
        with (
            patch(
                "src.modules.exports.adapters.inbound.job_control.export_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda func, *args: func(*args)),
            ),
            patch.object(
                Path,
                "exists",
                autospec=True,
                side_effect=lambda path: str(path).endswith("best_cer.pt"),
            ),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-4",
                    event_type="vit_ctc_deepseek",
                    payload={"checkpoint_best_name": "best.pt"},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        spec = self.recognizer_export.execute.call_args.args[0]
        self.assertEqual(
            spec.checkpoint_path,
            Path("/tmp/export-job-handler-test/artifacts/recognizer/best_cer.pt"),
        )

    async def test_handle_returns_failed_result_for_unknown_event_type(self) -> None:
        result = await self.handler.handle(
            ClaimedJobDto(
                request_id="req-3",
                event_type="unknown",
                payload={},
                status="PROCESSING",
            )
        )

        self.assertFalse(result.success)
        self.assertIn("Unsupported export event_type", result.error_message or "")
        self.detection_export.execute.assert_not_called()
        self.recognizer_export.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
