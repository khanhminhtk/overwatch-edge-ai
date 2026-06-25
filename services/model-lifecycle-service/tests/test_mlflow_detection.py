from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.detection.mlflow import MLflowDetection  # noqa: E402


class MLflowDetectionTest(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "MLFLOW_MODEL_NAME_DETECTION": "YoloTextDetector",
            "DETECT_MODEL_TYPE": "detector_training",
            "GPU_NAME": "RTX-4090",
        },
        clear=False,
    )
    def test_execute_runs_detection_flow_then_finishes_run(self) -> None:
        tracking_detection = Mock()
        artifacts = Mock()
        model_registry = Mock()
        tracking = Mock()
        logger = Mock()
        tracking.start_run.return_value = "run-det-1"

        use_case = MLflowDetection(
            mlflow_tracking_detection=tracking_detection,
            mlflow_artifacts=artifacts,
            mlflow_model_registry=model_registry,
            mlflow_tracking=tracking,
            logger=logger,
        )

        use_case.execute(
            git_commit="abc123",
            checkpoint_name=["best.pt", "last.pt"],
            goal="maximize_map",
            tracking_level="full",
        )

        tracking.start_run.assert_called_once_with(
            experiment_name="detector_training",
            run_name="detector_YoloTextDetector_abc123_RTX-4090",
        )
        artifacts.execute.assert_called_once_with(
            checkpoint_name=["best.pt", "last.pt"],
            run_id="run-det-1",
            pwd="",
        )
        tracking_detection.execute.assert_called_once_with(
            git_commit="abc123",
            goal="maximize_map",
            tracking_level="full",
            run_id="run-det-1",
            pwd="",
        )
        model_registry.execute.assert_called_once_with(
            run_id="run-det-1",
            git_commit="abc123",
            pwd="",
        )
        tracking.end_run.assert_called_once_with("run-det-1", status="FINISHED")
        logger.info.assert_has_calls(
            [
                call("[MLFLOW_WORKFLOW_RUN_STARTED]", "run_id=run-det-1", "model_name=YoloTextDetector"),
                call("[MLFLOW_WORKFLOW_ARTIFACTS_UPLOADED]", "run_id=run-det-1"),
                call("[MLFLOW_WORKFLOW_TRACKING_LOGGED]", "run_id=run-det-1"),
                call("[MLFLOW_WORKFLOW_MODEL_REGISTERED]", "run_id=run-det-1"),
                call("[MLFLOW_WORKFLOW_RUN_FINISHED]", "run_id=run-det-1", "status=FINISHED"),
            ]
        )
