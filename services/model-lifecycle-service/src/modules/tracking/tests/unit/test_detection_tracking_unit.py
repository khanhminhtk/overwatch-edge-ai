from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.use_case.detection.tracking import (  # noqa: E402
    DetectionMlflowTracking,
)
from src.modules.tracking.domain.entity_objects import DetectionRun  # noqa: E402
from src.modules.tracking.domain.value_objects import DetectionConfig  # noqa: E402
from src.modules.tracking.domain.value_objects import HardwareInfo  # noqa: E402


class DetectionMlflowTrackingUnitTest(unittest.TestCase):
    def test_execute_sets_source_and_git_commit_tags(self) -> None:
        tracker = Mock()
        trace_port = Mock()
        logger = Mock()
        run = DetectionRun(
            experiment_name="exp",
            run_name="run",
            config=DetectionConfig(
                model_name="yolo_detector",
                deployment_stage="training",
                hardware=HardwareInfo(gpu_host_name="edge-host", gpu_name="RTX-4090"),
            ),
        )
        use_case = DetectionMlflowTracking(
            run=run,
            hardware_info=run.config.hardware,
            experiment_tracker=tracker,
            trace_port=trace_port,
            logger=logger,
        )

        with patch(
            "src.modules.tracking.application.use_case.detection.tracking.get_commit_hash",
            return_value="abc123",
        ), patch.object(use_case, "export_training_metrics"):
            use_case.execute(run_id="run-123", pwd="/repo")

        tracker.resume_run.assert_called_once_with("run-123")
        tracker.set_tags.assert_called_once_with(
            run_id="run-123",
            tags={
                MLFLOW_SOURCE_NAME: "services/model-lifecycle-service/src/modules/tracking/tests/integration/detection_main.py",
                MLFLOW_SOURCE_TYPE: "LOCAL",
                "model_type": "detector",
                "task": "detection",
                "registered_model_name": "yolo_detector",
                "deployment_stage": "training",
                MLFLOW_GIT_COMMIT: "abc123",
                "host_name": "edge-host",
                "gpu_name": "RTX-4090",
                "framework": "unknown_framework",
            },
        )

    def test_execute_logs_exception_before_reraising(self) -> None:
        tracker = Mock()
        logger = Mock()
        use_case = DetectionMlflowTracking(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            hardware_info=HardwareInfo(),
            experiment_tracker=tracker,
            trace_port=Mock(),
            logger=logger,
        )

        with patch(
            "src.modules.tracking.application.use_case.detection.tracking.get_commit_hash",
            return_value="abc123",
        ), patch.object(
            use_case,
            "export_training_metrics",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                use_case.execute(run_id="run-123", pwd="/repo")

        logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
