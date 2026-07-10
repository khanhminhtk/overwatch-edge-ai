from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.use_case.detection.workflow import (  # noqa: E402
    DetectionMlflowWorkflow,
)


class DetectionMlflowWorkflowUnitTest(unittest.TestCase):
    def test_execute_runs_tracking_artifacts_registry_in_order(self) -> None:
        tracking = Mock()
        tracking.start_run.return_value = "run-123"
        artifacts = Mock()
        registry = Mock()
        registry.execute.return_value = "12"

        workflow = DetectionMlflowWorkflow(
            tracking=tracking,
            artifacts=artifacts,
            registry=registry,
        )

        model_version = workflow.execute(
            checkpoint_names=["best.pt", "last.pt"],
            pwd="/repo",
        )

        self.assertEqual(model_version, "12")
        tracking.start_run.assert_called_once_with()
        tracking.execute.assert_called_once_with(run_id="run-123", pwd="/repo")
        artifacts.execute.assert_called_once_with(
            checkpoint_name=["best.pt", "last.pt"],
            run_id="run-123",
            pwd="/repo",
        )
        registry.execute.assert_called_once_with(run_id="run-123", pwd="/repo")
        tracking.end_run.assert_called_once_with(run_id="run-123")

    def test_execute_marks_run_failed_when_registry_errors(self) -> None:
        tracking = Mock()
        tracking.start_run.return_value = "run-123"
        artifacts = Mock()
        registry = Mock()
        registry.execute.side_effect = RuntimeError("boom")

        workflow = DetectionMlflowWorkflow(
            tracking=tracking,
            artifacts=artifacts,
            registry=registry,
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            workflow.execute(
                checkpoint_names=["best.pt", "last.pt"],
                pwd="/repo",
            )

        tracking.end_run.assert_called_once_with(run_id="run-123", status="FAILED")


if __name__ == "__main__":
    unittest.main()
