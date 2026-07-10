from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.adapters.inbound.kafka.ingest_job_handler import (  # noqa: E402
    IngestJobHandler,
)
from src.platform.messaging.kafka.message import ConsumedMessage  # noqa: E402


class IngestJobHandlerUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.ingest_job = MagicMock()
        self.ingest_job.execute = AsyncMock(return_value=True)
        self.logger = MagicMock()
        self.handler = IngestJobHandler(
            ingest_job=self.ingest_job,
            logger=self.logger,
            consumer_group="mlflow-worker-detection",
        )

    async def test_handle_creates_job_record_from_kafka_message(self) -> None:
        message = ConsumedMessage(
            topic="model.lifecycle.mlflow.tracking",
            partition=1,
            offset=99,
            key=b"yolo_detector",
            value=json.dumps(
                {
                    "request_id": "req-1",
                    "event_type": "yolo_detector",
                    "payload": {
                        "model_name": "yolo_detector",
                        "git_commit": "abc123",
                    },
                }
            ).encode("utf-8"),
        )

        created = await self.handler.handle(message)

        self.assertTrue(created)
        call = self.ingest_job.execute.await_args
        self.assertEqual(call.args[0].request_id, "req-1")
        self.assertEqual(call.args[0].event_type, "yolo_detector")
        self.assertEqual(call.args[0].schema_name, "mlflow_tracking_job")
        self.assertEqual(call.args[0].schema_version, "1.0")
        self.assertEqual(call.args[0].message_key, "yolo_detector")
        self.assertEqual(
            call.args[0].message_identity.consumer_group,
            "mlflow-worker-detection",
        )

    async def test_handle_propagates_ingest_validation_errors(self) -> None:
        self.ingest_job.execute = AsyncMock(
            side_effect=ValueError(
                "Unexpected event_type=vit_ctc_deepseek; expected yolo_detector"
            )
        )
        message = ConsumedMessage(
            topic="model.lifecycle.mlflow.tracking",
            partition=1,
            offset=99,
            key=None,
            value=json.dumps(
                {
                    "request_id": "req-1",
                    "event_type": "vit_ctc_deepseek",
                    "payload": {"model_name": "vit_ctc_deepseek"},
                }
            ).encode("utf-8"),
        )

        with self.assertRaisesRegex(ValueError, "Unexpected event_type"):
            await self.handler.handle(message)


if __name__ == "__main__":
    unittest.main()
