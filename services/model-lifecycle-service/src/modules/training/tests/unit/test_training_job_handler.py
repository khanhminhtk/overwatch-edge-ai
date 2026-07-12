from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.training.adapters.inbound.job_control.training_job_handler import (
    TrainingJobHandler,
)
from src.platform.logger import Logger


class TrainingJobHandlerUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.recognizer_training = MagicMock()
        self.detection_training = MagicMock()
        self.logger = MagicMock(spec=Logger)
        self.handler = TrainingJobHandler(
            recognizer_training=self.recognizer_training,
            detection_training=self.detection_training,
            recognizer_event_type="train_requested_recognizer",
            detection_event_type="train_requested_detection",
            default_mode="local",
            logger=self.logger,
        )

    async def test_handle_recognizer_job_by_event_type(self) -> None:
        with patch(
            "src.modules.training.adapters.inbound.job_control.training_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-1",
                    event_type="train_requested_recognizer",
                    payload={"dataset_version": "v1", "mode": "dev"},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        self.recognizer_training.execute.assert_called_once()
        spec = self.recognizer_training.execute.call_args.args[0]
        self.assertEqual(spec.model_name, "recognizer")
        self.assertEqual(spec.dataset_version, "v1")
        self.assertEqual(spec.mode, "dev")
        self.detection_training.execute.assert_not_called()

    async def test_handle_detection_job_by_event_type(self) -> None:
        with patch(
            "src.modules.training.adapters.inbound.job_control.training_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-2",
                    event_type="train_requested_detection",
                    payload={"dataset_version": "v2"},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        self.detection_training.execute.assert_called_once()
        spec = self.detection_training.execute.call_args.args[0]
        self.assertEqual(spec.model_name, "detector")
        self.assertEqual(spec.dataset_version, "v2")
        self.assertEqual(spec.mode, "local")
        self.recognizer_training.execute.assert_not_called()

    async def test_handle_uses_default_mode_when_not_in_payload(self) -> None:
        with patch(
            "src.modules.training.adapters.inbound.job_control.training_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-3",
                    event_type="train_requested_recognizer",
                    payload={"dataset_version": "v3"},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        spec = self.recognizer_training.execute.call_args.args[0]
        self.assertEqual(spec.mode, "local")

    async def test_handle_returns_failed_for_unknown_event_type(self) -> None:
        result = await self.handler.handle(
            ClaimedJobDto(
                request_id="req-4",
                event_type="unknown_event_type",
                payload={"dataset_version": "v4"},
                status="PROCESSING",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("Unsupported training event_type", result.error_message or "")

    async def test_handle_returns_failed_when_use_case_raises(self) -> None:
        self.recognizer_training.execute.side_effect = RuntimeError("training failed")
        with patch(
            "src.modules.training.adapters.inbound.job_control.training_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ):
            result = await self.handler.handle(
                ClaimedJobDto(
                    request_id="req-5",
                    event_type="train_requested_recognizer",
                    payload={"dataset_version": "v5"},
                    status="PROCESSING",
                )
            )
        self.assertFalse(result.success)
        self.assertIn("training failed", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
