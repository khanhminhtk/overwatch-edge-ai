from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.job_control.application.dto.consume_event_command import (  # noqa: E402
    ConsumeEventCommand,
)
from src.modules.job_control.application.dto.job_result_dto import JobResultDto  # noqa: E402
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity  # noqa: E402


class JobControlDtosUnitTest(unittest.TestCase):
    def test_consume_event_command_stores_ingest_inputs(self) -> None:
        identity = MessageIdentity(
            topic="model.lifecycle.mlflow.tracking",
            partition_id=1,
            message_offset=9,
            consumer_group="mlflow-worker-detection",
        )

        command = ConsumeEventCommand(
            request_id="req-1",
            event_type="yolo_detector",
            schema_name="mlflow_tracking_job",
            schema_version="1.0",
            message_key="yolo_detector",
            message_identity=identity,
            payload={"model_name": "yolo_detector"},
        )

        self.assertEqual(command.request_id, "req-1")
        self.assertEqual(command.event_type, "yolo_detector")
        self.assertIs(command.message_identity, identity)
        self.assertEqual(command.payload["model_name"], "yolo_detector")

    def test_claimed_job_dto_stores_claimed_projection(self) -> None:
        claimed = ClaimedJobDto(
            request_id="req-2",
            event_type="vit_ctc_deepseek",
            payload={"git_commit": "abc123"},
            status="PROCESSING",
        )

        self.assertEqual(claimed.request_id, "req-2")
        self.assertEqual(claimed.event_type, "vit_ctc_deepseek")
        self.assertEqual(claimed.status, "PROCESSING")

    def test_job_result_dto_tracks_completion_state(self) -> None:
        result = JobResultDto(
            request_id="req-3",
            success=False,
            error_message="tracking failed",
        )

        self.assertEqual(result.request_id, "req-3")
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "tracking failed")


if __name__ == "__main__":
    unittest.main()
