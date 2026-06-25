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

from src.applications.use_cases.mlflow.recognizer.mlflow_tracking import (  # noqa: E402
    MlflowTrackingRecognizer,
)


class MlflowTrackingRecognizerTest(unittest.TestCase):
    def test_log_tag_builds_full_tag_payload_and_passes_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "RECOG_TENSORBOARD_DIR=data/checkpoint_recognizer/runs/recognizer",
                        "RECOG_SAVE_DIR=data/checkpoint_recognizer",
                        "RECOG_BEST_CHECKPOINT_NAME=best.pt",
                        "RECOG_DATASET_ZIP_PATH=data/datasets/recognizer.zip",
                        "RECOG_MANIFEST_PATH=data/datasets/recognizer_manifest.json",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "RECOG_DOCKER_IMAGE=recognizer:latest",
                    ]
                ),
                encoding="utf-8",
            )

            tracking = Mock()
            logger = Mock()
            recognizer = MlflowTrackingRecognizer(
                mlflow_tracking=tracking,
                logger=logger,
                path_env_ml_traning=str(env_path),
            )

            recognizer.log_tag(
                git_commit="abc123",
                host_name="edge-host",
                gpu_name="RTX-4090",
                framework="pytorch",
                goal="accuracy",
                tracking_level="full",
                run_id="run-123",
            )

        tracking.resume_run.assert_called_once_with(run_id="run-123")
        tracking.set_tags.assert_called_once()
        self.assertEqual(tracking.set_tags.call_args.kwargs["run_id"], "run-123")
        tags = tracking.set_tags.call_args.kwargs["tags"]
        self.assertEqual(tags["deployment_stage"], "staging")
        self.assertEqual(tags["docker_image"], "recognizer:latest")
        self.assertEqual(tags["tensorboard_dir"], "data/checkpoint_recognizer/runs/recognizer")
        self.assertEqual(tags["checkpoint_dir"], "data/checkpoint_recognizer")
        self.assertEqual(tags["dataset_zip_path"], "data/datasets/recognizer.zip")
        self.assertEqual(tags["manifest_path"], "data/datasets/recognizer_manifest.json")
        self.assertEqual(tags["metadata_mocked"], "false")
        tracking.end_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
