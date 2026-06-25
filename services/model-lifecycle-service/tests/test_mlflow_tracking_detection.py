from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.detection.mlflow_tracking import (  # noqa: E402
    MlflowTrackingDetection,
)


class MlflowTrackingDetectionTest(unittest.TestCase):
    def test_log_tag_uses_detection_env_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DETECT_RESULTS_CSV_PATH=artifacts/detection/yolo/results.csv",
                        "DETECT_SAVE_DIR=artifacts/detection/yolo/weights",
                        "DETECT_DATASET_ZIP_PATH=ml/training/data/detection.zip",
                        "DETECT_MANIFEST_PATH=ml/training/data/detection_dataset_manifest.json",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "DETECT_DOCKER_IMAGE=detector:latest",
                        "MLFLOW_MODEL_NAME_DETECTION=YoloTextDetector",
                        "DETECT_TASK_NAME=object_detection",
                        "DETECT_MODEL_TYPE=detection",
                    ]
                ),
                encoding="utf-8",
            )

            tracking = Mock()
            recognizer = MlflowTrackingDetection(
                mlflow_tracking=tracking,
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )

            recognizer.log_tag(
                git_commit="abc123",
                host_name="edge-host",
                gpu_name="RTX-4090",
                framework="pytorch",
                goal="map",
                tracking_level="full",
                run_id="run-123",
            )

        tags = tracking.set_tags.call_args.kwargs["tags"]
        self.assertEqual(tags["registered_model_name"], "YoloTextDetector")
        self.assertEqual(tags["task"], "object_detection")
        self.assertEqual(tags["tensorboard_dir"], "artifacts/detection/yolo/results.csv")
        self.assertEqual(tags["checkpoint_dir"], "artifacts/detection/yolo/weights")
        self.assertEqual(tags["dataset_zip_path"], "ml/training/data/detection.zip")
