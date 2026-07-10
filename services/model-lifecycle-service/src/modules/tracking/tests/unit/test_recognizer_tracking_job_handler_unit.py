from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.tracking.adapters.inbound.job_control.tracking_job_handler import (  # noqa: E402
    RecognizerTrackingJobHandler,
)


class RecognizerTrackingJobHandlerUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_handle_executes_workflow_and_returns_success_result(self) -> None:
        workflow = Mock()
        workflow.execute.return_value = "12"
        handler = RecognizerTrackingJobHandler(
            workflow=workflow,
            pwd="/repo",
            default_checkpoint_names=["best_cer.pt", "last_checkpoint.pt"],
        )

        with patch(
            "src.modules.tracking.adapters.inbound.job_control.tracking_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, **kwargs: fn(**kwargs)),
        ) as to_thread_mock:
            result = await handler.handle(
                ClaimedJobDto(
                    request_id="req-1",
                    event_type="vit_ctc_deepseek",
                    payload={
                        "checkpoint_best_name": "best_custom.pt",
                        "checkpoint_last_name": "last_custom.pt",
                    },
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.request_id, "req-1")
        workflow.execute.assert_called_once_with(
            checkpoint_names=["best_custom.pt", "last_custom.pt"],
            pwd="/repo",
        )
        to_thread_mock.assert_awaited_once()

    async def test_handle_runs_workflow_in_background_thread(self) -> None:
        workflow = Mock()
        workflow.execute.return_value = "12"
        handler = RecognizerTrackingJobHandler(
            workflow=workflow,
            pwd="/repo",
            default_checkpoint_names=["best_cer.pt", "last_checkpoint.pt"],
        )

        with patch(
            "src.modules.tracking.adapters.inbound.job_control.tracking_job_handler.asyncio.to_thread",
            new=AsyncMock(return_value="12"),
        ) as to_thread_mock:
            result = await handler.handle(
                ClaimedJobDto(
                    request_id="req-thread",
                    event_type="vit_ctc_deepseek",
                    payload={
                        "checkpoint_best_name": "best_custom.pt",
                        "checkpoint_last_name": "last_custom.pt",
                    },
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        to_thread_mock.assert_awaited_once_with(
            workflow.execute,
            checkpoint_names=["best_custom.pt", "last_custom.pt"],
            pwd="/repo",
        )

    async def test_handle_uses_default_checkpoint_names_when_payload_missing(self) -> None:
        workflow = Mock()
        workflow.execute.return_value = "13"
        handler = RecognizerTrackingJobHandler(
            workflow=workflow,
            pwd="/repo",
            default_checkpoint_names=["best_cer.pt", "last_checkpoint.pt"],
        )

        with patch(
            "src.modules.tracking.adapters.inbound.job_control.tracking_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, **kwargs: fn(**kwargs)),
        ) as to_thread_mock:
            result = await handler.handle(
                ClaimedJobDto(
                    request_id="req-2",
                    event_type="vit_ctc_deepseek",
                    payload={},
                    status="PROCESSING",
                )
            )

        self.assertTrue(result.success)
        workflow.execute.assert_called_once_with(
            checkpoint_names=["best_cer.pt", "last_checkpoint.pt"],
            pwd="/repo",
        )
        to_thread_mock.assert_awaited_once()

    async def test_handle_returns_failed_result_when_workflow_raises(self) -> None:
        workflow = Mock()
        workflow.execute.side_effect = RuntimeError("mlflow exploded")
        handler = RecognizerTrackingJobHandler(
            workflow=workflow,
            pwd="/repo",
            default_checkpoint_names=["best_cer.pt", "last_checkpoint.pt"],
        )

        with patch(
            "src.modules.tracking.adapters.inbound.job_control.tracking_job_handler.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("mlflow exploded")),
        ):
            result = await handler.handle(
                ClaimedJobDto(
                    request_id="req-3",
                    event_type="vit_ctc_deepseek",
                    payload={},
                    status="PROCESSING",
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.request_id, "req-3")
        self.assertIn("mlflow exploded", result.error_message or "")

if __name__ == "__main__":
    unittest.main()
