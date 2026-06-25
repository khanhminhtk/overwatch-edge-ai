from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.detection.mlflow_model_registry import (  # noqa: E402
    MlflowModelRegistryDetection,
)


class MlflowModelRegistryDetectionTest(unittest.TestCase):
    def test_execute_registers_detection_model_with_detection_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MLFLOW_MODEL_NAME_DETECTION=YoloTextDetector",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "DETECT_BEST_CHECKPOINT_NAME=best.pt",
                        f"DETECT_SAVE_DIR={temp_dir}",
                        "DETECT_DATASET_NAME=detection_dataset",
                        "PIPELINE_NAME_DETECTION=detector_yolo",
                        "PIPELINE_RUN_ID_DETECTION=detector_yolo_001",
                    ]
                ),
                encoding="utf-8",
            )

            registry = Mock()
            registry.register_model.return_value = "3"
            artifact = Mock()
            use_case = MlflowModelRegistryDetection(
                mlflow_model_registry=registry,
                mlflow_artifact=artifact,
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )

            with patch.object(
                MlflowModelRegistryDetection,
                "_load_model_metadata",
                return_value={
                    "epoch": 10,
                    "best_fitness": 0.91,
                    "version": "8.3.0",
                    "metrics/precision(B)": 0.67192,
                    "metrics/recall(B)": 0.62359,
                    "metrics/mAP50(B)": 0.66829,
                    "metrics/mAP50-95(B)": 0.41476,
                    "val/box_loss": 1.4435,
                    "val/cls_loss": 1.22764,
                    "val/dfl_loss": 0.00384,
                    "fitness": 0.41476,
                    "model_total_params": 100,
                },
            ):
                version = use_case.execute(run_id="run-det-1", git_commit="abc123", framework="pytorch")

        self.assertEqual(version, "3")
        tags = registry.set_model_version_tags.call_args.kwargs["tags"]
        self.assertEqual(tags["model_type"], "detection")
        self.assertEqual(tags["checkpoint_epoch"], "10")
        self.assertEqual(tags["best_fitness"], "0.91")
        self.assertEqual(tags["checkpoint_version"], "8.3.0")
        self.assertEqual(tags["metrics_precision_b"], "0.67192")
        self.assertEqual(tags["metrics_recall_b"], "0.62359")
        self.assertEqual(tags["metrics_map50_b"], "0.66829")
        self.assertEqual(tags["metrics_map50_95_b"], "0.41476")
        self.assertEqual(tags["val_box_loss"], "1.4435")
        self.assertEqual(tags["val_cls_loss"], "1.22764")
        self.assertEqual(tags["val_dfl_loss"], "0.00384")
        self.assertEqual(tags["fitness"], "0.41476")
