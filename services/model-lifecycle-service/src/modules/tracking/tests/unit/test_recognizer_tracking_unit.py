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

from src.modules.tracking.application.use_case.recognizer.tracking import (  # noqa: E402
    RecognizerMlflowTracking,
)
from src.modules.tracking.domain.entity_objects import RecognizerRun  # noqa: E402
from src.modules.tracking.domain.value_objects import HardwareInfo, RecognizerConfig  # noqa: E402


class RecognizerMlflowTrackingUnitTest(unittest.TestCase):
    def test_execute_sets_source_and_git_commit_tags(self) -> None:
        tracker = Mock()
        use_case = RecognizerMlflowTracking(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(
                    model_name="vit_ctc_deepseek",
                    deployment_stage="training",
                    hardware=HardwareInfo(gpu_host_name="edge-host", gpu_name="RTX-4090"),
                ),
            ),
            hardware_info=HardwareInfo(gpu_host_name="edge-host", gpu_name="RTX-4090"),
            experiment_tracker=tracker,
            trace_port=Mock(),
            logger=Mock(),
        )

        with patch(
            "src.modules.tracking.application.use_case.recognizer.tracking.get_commit_hash",
            return_value="abc123",
        ), patch.object(use_case, "export_training_metrics"):
            use_case.execute(run_id="run-123", pwd="/repo")

        tracker.resume_run.assert_called_once_with("run-123")
        tracker.set_tags.assert_called_once_with(
            run_id="run-123",
            tags={
                MLFLOW_SOURCE_NAME: "services/model-lifecycle-service/src/modules/tracking/tests/integration/recognizer_main.py",
                MLFLOW_SOURCE_TYPE: "LOCAL",
                "model_type": "recognizer",
                "task": "recognition",
                "registered_model_name": "vit_ctc_deepseek",
                "deployment_stage": "training",
                MLFLOW_GIT_COMMIT: "abc123",
                "host_name": "edge-host",
                "gpu_name": "RTX-4090",
                "framework": "unknown_framework",
            },
        )

    def test_export_training_metrics_skips_when_tensorboard_dir_missing(self) -> None:
        logger = Mock()
        use_case = RecognizerMlflowTracking(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(
                    model_name="vit_ctc_deepseek",
                    tensorboard_dir="missing/tensorboard",
                    hardware=HardwareInfo(gpu_host_name="edge-host", gpu_name="RTX-4090"),
                ),
            ),
            hardware_info=HardwareInfo(gpu_host_name="edge-host", gpu_name="RTX-4090"),
            experiment_tracker=Mock(),
            trace_port=Mock(),
            logger=logger,
        )

        use_case.export_training_metrics(run_id="run-123", pwd="/repo")

        logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
