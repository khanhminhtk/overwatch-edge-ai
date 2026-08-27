from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.tracking.adapters.inbound.job_control.download_job_handler import (  # noqa: E402
    DownloadJobHandler,
)


class DownloadJobHandlerUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_handle_downloads_champion_checkpoint_when_version_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detection_download = Mock()
            recognizer_download = Mock()
            handler = DownloadJobHandler(
                detection_download=detection_download,
                recognizer_download=recognizer_download,
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
            ) as to_thread_mock:
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-1",
                        event_type="yolo_detector",
                        payload={
                            "output_path": "data/download/yolo_detector/best.pt",
                        },
                        status="PROCESSING",
                    )
                )
            self.assertTrue(result.success)
            detection_download.download_champion_to_file.assert_called_once_with(
                "checkpoints/best.pt",
                f"{tmpdir}/data/download/yolo_detector/best.pt",
            )
            recognizer_download.download_champion_to_file.assert_not_called()
            to_thread_mock.assert_awaited_once()

    async def test_handle_downloads_requested_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detection_download = Mock()
            recognizer_download = Mock()
            handler = DownloadJobHandler(
                detection_download=detection_download,
                recognizer_download=recognizer_download,
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-2",
                        event_type="vit_ctc_deepseek",
                        payload={
                            "checkpoint_best_name": "custom.pt",
                            "model_version": "17",
                            "output_path": "data/download/vit_ctc_deepseek/custom.pt",
                        },
                        status="PROCESSING",
                    )
                )
            self.assertTrue(result.success)
            recognizer_download.download_version_to_file.assert_called_once_with(
                "17",
                "checkpoints/custom.pt",
                f"{tmpdir}/data/download/vit_ctc_deepseek/custom.pt",
            )
            detection_download.download_version_to_file.assert_not_called()

    async def test_handle_returns_failure_for_unsupported_event_type(self) -> None:
        handler = DownloadJobHandler(
            detection_download=Mock(),
            recognizer_download=Mock(),
            detection_event_type="yolo_detector",
            recognizer_event_type="vit_ctc_deepseek",
            detection_default_checkpoint_name="best.pt",
            recognizer_default_checkpoint_name="best_cer.pt",
            pwd="/repo",
        )

        result = await handler.handle(
            ClaimedJobDto(
                request_id="req-3",
                event_type="unknown_model",
                payload={},
                status="PROCESSING",
            )
        )

        self.assertFalse(result.success)
        self.assertIn("Unsupported download event_type", result.error_message or "")

    async def test_handle_returns_failure_when_output_path_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer_download = Mock()
            handler = DownloadJobHandler(
                detection_download=Mock(),
                recognizer_download=recognizer_download,
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-missing-output-path",
                        event_type="vit_ctc_deepseek",
                        payload={},
                        status="PROCESSING",
                    )
                )

            self.assertFalse(result.success)
            self.assertIn("payload.output_path", result.error_message or "")
            recognizer_download.download_champion_to_file.assert_not_called()

    async def test_handle_defaults_download_to_configured_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detection_download = Mock()
            handler = DownloadJobHandler(
                detection_download=detection_download,
                recognizer_download=Mock(),
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                detection_checkpoint_dir="data/checkpoint/yolo/yolo_finetune/weights",
                recognizer_checkpoint_dir="data/checkpoint_recognizer",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-default-output",
                        event_type="yolo_detector",
                        payload={},
                        status="PROCESSING",
                    )
                )

            self.assertTrue(result.success)
            detection_download.download_champion_to_file.assert_called_once_with(
                "checkpoints/best.pt",
                f"{tmpdir}/data/checkpoint/yolo/yolo_finetune/weights/best.pt",
            )

    async def test_handle_uses_output_path_from_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detection_download = Mock()
            handler = DownloadJobHandler(
                detection_download=detection_download,
                recognizer_download=Mock(),
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-output-path",
                        event_type="yolo_detector",
                        payload={
                            "output_path": "data/test-downloads/detection/manual-best.pt",
                        },
                        status="PROCESSING",
                    )
                )

            self.assertTrue(result.success)
            detection_download.download_champion_to_file.assert_called_once_with(
                "checkpoints/best.pt",
                f"{tmpdir}/data/test-downloads/detection/manual-best.pt",
            )

    async def test_handle_returns_failure_when_download_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            detection_download = Mock()
            detection_download.download_champion_to_file.side_effect = RuntimeError("download failed")
            handler = DownloadJobHandler(
                detection_download=detection_download,
                recognizer_download=Mock(),
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
            )

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.to_thread",
                new=AsyncMock(side_effect=RuntimeError("download failed")),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-4",
                        event_type="yolo_detector",
                        payload={
                            "output_path": "data/download/yolo_detector/best.pt",
                        },
                        status="PROCESSING",
                    )
                )
            self.assertFalse(result.success)
            self.assertEqual(result.request_id, "req-4")
            self.assertIn("download failed", result.error_message or "")

    async def test_handle_returns_failure_when_download_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = DownloadJobHandler(
                detection_download=Mock(),
                recognizer_download=Mock(),
                detection_event_type="yolo_detector",
                recognizer_event_type="vit_ctc_deepseek",
                detection_default_checkpoint_name="best.pt",
                recognizer_default_checkpoint_name="best_cer.pt",
                pwd=tmpdir,
                download_timeout_seconds=5.0,
            )

            async def _raise_timeout(awaitable, timeout):
                self.assertEqual(timeout, 5.0)
                awaitable.close()
                raise asyncio.TimeoutError()

            with patch(
                "src.modules.tracking.adapters.inbound.job_control.download_job_handler.asyncio.wait_for",
                new=AsyncMock(side_effect=_raise_timeout),
            ):
                result = await handler.handle(
                    ClaimedJobDto(
                        request_id="req-timeout",
                        event_type="yolo_detector",
                        payload={
                            "output_path": "data/download/yolo_detector/best.pt",
                        },
                        status="PROCESSING",
                    )
                )

            self.assertFalse(result.success)
            self.assertIn("timed out", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
