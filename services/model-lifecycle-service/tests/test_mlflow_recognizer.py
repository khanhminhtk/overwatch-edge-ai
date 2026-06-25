from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.recognizer.mlflow import MLflowRecognizer  # noqa: E402


class MLflowRecognizerTest(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "MLFLOW_MODEL_NAME_RECOGNIZER": "VitCTCDeepSeek",
            "RECOG_MODEL_TYPE": "recognizer_training",
            "GPU_NAME": "RTX-4090",
            "FRAMEWORK_NAME": "pytorch",
        },
        clear=False,
    )
    def test_execute_runs_tracking_artifacts_and_registry_then_finishes_run(self) -> None:
        tracking_recognizer = Mock()
        artifacts = Mock()
        model_registry = Mock()
        tracking = Mock()
        logger = Mock()
        tracking.start_run.return_value = "run-123"

        recognizer = MLflowRecognizer(
            mlflow_tracking_recognizer=tracking_recognizer,
            mlflow_artifacts=artifacts,
            mlflow_model_registry=model_registry,
            mlflow_tracking=tracking,
            logger=logger,
        )

        recognizer.execute(
            git_commit="abc123",
            checkpoint_name=["best_cer.pt", "last_checkpoint.pt"],
            goal="maximize_accuracy",
            tracking_level="full",
        )

        tracking.start_run.assert_called_once_with(
            experiment_name="recognizer_training",
            run_name="recognizer_VitCTCDeepSeek_abc123_RTX-4090",
        )
        tracking.resume_run.assert_called_once_with("run-123")
        artifacts.execute.assert_called_once_with(
            checkpoint_name=["best_cer.pt", "last_checkpoint.pt"],
            run_id="run-123",
            pwd="",
        )
        tracking_recognizer.execute.assert_called_once_with(
            git_commit="abc123",
            goal="maximize_accuracy",
            tracking_level="full",
            run_id="run-123",
            pwd="",
        )
        model_registry.execute.assert_called_once_with(
            run_id="run-123",
            git_commit="abc123",
            pwd="",
        )
        tracking.end_run.assert_called_once_with("run-123", status="FINISHED")
        logger.info.assert_has_calls(
            [
                call("[MLFLOW_WORKFLOW_RUN_STARTED]", "run_id=run-123", "model_name=VitCTCDeepSeek"),
                call("[MLFLOW_WORKFLOW_ARTIFACTS_UPLOADED]", "run_id=run-123"),
                call("[MLFLOW_WORKFLOW_TRACKING_LOGGED]", "run_id=run-123"),
                call("[MLFLOW_WORKFLOW_MODEL_REGISTERED]", "run_id=run-123"),
                call("[MLFLOW_WORKFLOW_RUN_FINISHED]", "run_id=run-123", "status=FINISHED"),
            ]
        )
        logger.exception.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            "MLFLOW_MODEL_NAME_RECOGNIZER": "VitCTCDeepSeek",
            "RECOG_MODEL_TYPE": "recognizer_training",
            "GPU_NAME": "RTX-4090",
            "FRAMEWORK_NAME": "pytorch",
        },
        clear=False,
    )
    def test_execute_marks_run_failed_and_reraises_when_step_errors(self) -> None:
        tracking_recognizer = Mock()
        artifacts = Mock()
        artifacts.execute.side_effect = RuntimeError("artifact upload failed")
        model_registry = Mock()
        tracking = Mock()
        logger = Mock()
        tracking.start_run.return_value = "run-456"

        recognizer = MLflowRecognizer(
            mlflow_tracking_recognizer=tracking_recognizer,
            mlflow_artifacts=artifacts,
            mlflow_model_registry=model_registry,
            mlflow_tracking=tracking,
            logger=logger,
        )

        with self.assertRaisesRegex(RuntimeError, "artifact upload failed"):
            recognizer.execute(
                git_commit="def456",
                checkpoint_name=["best_cer.pt"],
                goal="maximize_accuracy",
                tracking_level="full",
            )

        tracking.end_run.assert_called_once_with("run-456", status="FAILED")
        tracking_recognizer.execute.assert_not_called()
        model_registry.execute.assert_not_called()
        logger.exception.assert_called_once_with(
            "[MLFLOW_WORKFLOW_RUN_FAILED]",
            "run_id=run-456",
            "error=artifact upload failed",
        )
        logger.info.assert_any_call(
            "[MLFLOW_WORKFLOW_RUN_FINISHED]",
            "run_id=run-456",
            "status=FAILED",
        )


if __name__ == "__main__":
    unittest.main()
