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

from src.applications.use_cases.mlflow.detection.mlflow_artifacts import (  # noqa: E402
    MlflowArtifactsDetection,
)


class MlflowArtifactsDetectionTest(unittest.TestCase):
    def test_upload_checkpoint_artifacts_logs_yolo_metrics_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"DETECT_SAVE_DIR={temp_dir}",
                        "MLFLOW_MODEL_NAME_DETECTION=YoloTextDetector",
                        "DETECT_BEST_CHECKPOINT_NAME=best.pt",
                        "DETECT_LAST_CHECKPOINT_NAME=last.pt",
                    ]
                ),
                encoding="utf-8",
            )

            tracking = Mock()
            use_case = MlflowArtifactsDetection(
                mlflow_tracking=tracking,
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )
            use_case.log_artifact = Mock()

            with patch.object(
                MlflowArtifactsDetection,
                "_load_checkpoint_metadata",
                side_effect=[
                    {
                        "epoch": 12,
                        "train_metrics": {
                            "metrics/precision(B)": 0.67,
                            "metrics/recall(B)": 0.62,
                            "metrics/mAP50(B)": 0.66,
                            "metrics/mAP50-95(B)": 0.41,
                            "fitness": 0.41,
                        },
                        "train_results": {
                            "epoch": [1, 2],
                            "metrics/mAP50(B)": [0.5, 0.66],
                            "val/box_loss": [1.6, 1.4],
                        },
                    },
                    {
                        "epoch": 20,
                        "train_metrics": {
                            "metrics/precision(B)": 0.70,
                            "metrics/recall(B)": 0.65,
                        },
                        "train_results": {
                            "epoch": [1, 2, 3],
                            "metrics/mAP50(B)": [0.5, 0.66, 0.7],
                        },
                    },
                ],
            ):
                use_case._upload_checkpoint_artifacts(
                    run_id="run-123",
                    checkpoint_name=["last.pt", "best.pt"],
                    pwd="",
                )

        self.assertEqual(use_case.log_artifact.call_count, 2)
        tracking.log_params.assert_any_call(
            "run-123",
            {
                "epoch": 12,
                "train_metrics.metrics_precision_b": 0.67,
                "train_metrics.metrics_recall_b": 0.62,
                "train_metrics.metrics_map50_b": 0.66,
                "train_metrics.metrics_map50_95_b": 0.41,
                "train_metrics.fitness": 0.41,
                "train_results_last.metrics_map50_b": 0.66,
                "train_results_last.val_box_loss": 1.4,
            },
        )
        tracking.log_params.assert_any_call(
            "run-123",
            {
                "epoch_best_map50": 20,
                "best_metrics.metrics_precision_b": 0.70,
                "best_metrics.metrics_recall_b": 0.65,
            },
        )


if __name__ == "__main__":
    unittest.main()
