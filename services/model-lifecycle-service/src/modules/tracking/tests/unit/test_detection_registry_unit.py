from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from mlflow.pyfunc import PythonModel


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.use_case.detection.registry import (  # noqa: E402
    DetectionMlflowModelRegistry,
)
from src.modules.tracking.domain.entity_objects import DetectionRun  # noqa: E402
from src.modules.tracking.domain.value_objects import DetectionConfig  # noqa: E402


class DetectionMlflowModelRegistryUnitTest(unittest.TestCase):
    def test_execute_logs_single_registered_pyfunc_model(self) -> None:
        use_case = DetectionMlflowModelRegistry(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            registry=Mock(),
            tracker=Mock(),
            logger=Mock(),
        )

        with patch.object(use_case, "log_model_version_info") as log_model_version_info:
            use_case.execute(run_id="run-123", pwd="/repo")

        log_model_version_info.assert_called_once_with(run_id="run-123", pwd="/repo")

    def test_log_model_version_info_passes_python_model_instead_of_checkpoint_path(self) -> None:
        registry = Mock()
        tracker = Mock()
        logger = Mock()
        run = DetectionRun(
            experiment_name="exp",
            run_name="run",
            config=DetectionConfig(
                model_name="yolo_detector",
                checkpoint_dir="data/checkpoint/yolo/weights",
                best_checkpoint_name="best.pt",
            ),
        )
        use_case = DetectionMlflowModelRegistry(
            run=run,
            registry=registry,
            tracker=tracker,
            logger=logger,
        )

        with patch.object(
            use_case,
            "_load_checkpoint",
            return_value={"epoch": 1, "train_metrics": {}},
        ):
            use_case.log_model_version_info(run_id="run-123", pwd="/repo")

        python_model = registry.log_pyfunc_model.call_args.kwargs["python_model"]
        self.assertIsInstance(python_model, PythonModel)
        self.assertEqual(
            python_model._checkpoint_path,
            "/repo/data/checkpoint/yolo/weights/best.pt",
        )

    def test_log_model_version_info_logs_input_signature_and_run_metadata(self) -> None:
        registry = Mock()
        tracker = Mock()
        logger = Mock()
        run = DetectionRun(
            experiment_name="exp",
            run_name="run",
            config=DetectionConfig(
                model_name="yolo_detector",
                checkpoint_dir="data/checkpoint/yolo/weights",
                best_checkpoint_name="best.pt",
            ),
        )
        use_case = DetectionMlflowModelRegistry(
            run=run,
            registry=registry,
            tracker=tracker,
            logger=logger,
        )
        checkpoint = {
            "epoch": 7,
            "train_metrics": {
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.6,
                "metrics/mAP50-95(B)": 0.5,
                "val/box_loss": 0.4,
                "val/dfl_loss": 0.3,
                "val/cls_loss": 0.2,
                "fitness": 0.1,
            },
        }

        with patch.object(
            use_case,
            "_load_checkpoint",
            return_value=checkpoint,
        ):
            use_case.log_model_version_info(run_id="run-123", pwd="/repo")

        kwargs = registry.log_pyfunc_model.call_args.kwargs
        self.assertIsInstance(kwargs["input_example"], pd.DataFrame)
        self.assertEqual(
            kwargs["input_example"].to_dict(orient="records"),
            [{"image_path": "sample.jpg", "request_id": "example-request"}],
        )
        self.assertIsNotNone(kwargs["signature"])
        tracker.log_params.assert_called_once_with(
            run_id="run-123",
            params={
                "hardware": "unknown_gpu",
                "epoch": 7,
            },
        )
        tracker.log_metrics.assert_called_once_with(
            run_id="run-123",
            metrics={
                "precision_b": 0.8,
                "recall_b": 0.7,
                "map50_b": 0.6,
                "map50_95_b": 0.5,
                "val_box_loss": 0.4,
                "val_dfl_loss": 0.3,
                "val_cls_loss": 0.2,
                "fitness": 0.1,
            },
        )

    def test_execute_promotes_when_map50_is_better_than_champion(self) -> None:
        registry = Mock()
        registry.log_pyfunc_model.return_value = "runs:/run-123/yolo_detector"
        registry.register_model.return_value = "12"
        registry.get_model_version_by_alias.return_value = Mock(tags={"map50": "0.5"})
        tracker = Mock()
        use_case = DetectionMlflowModelRegistry(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(
                    model_name="yolo_detector",
                    checkpoint_dir="data/checkpoint/yolo/weights",
                    best_checkpoint_name="best.pt",
                ),
            ),
            registry=registry,
            tracker=tracker,
            logger=Mock(),
        )

        with patch.object(
            use_case,
            "_load_checkpoint",
            return_value={"epoch": 1, "train_metrics": {"metrics/mAP50(B)": 0.6}},
        ):
            version = use_case.execute(run_id="run-123", pwd="/repo")

        self.assertEqual(version, "12")
        registry.set_model_version_tags.assert_called_once_with(
            registered_model_name="yolo_detector",
            model_version="12",
            tags={"map50": "0.6"},
        )
        registry.set_model_alias.assert_called_once_with("yolo_detector", "champion", "12")
        registry.transition_model_stage.assert_called_once_with("yolo_detector", "12", "Production")

    def test_execute_does_not_replace_champion_on_tie(self) -> None:
        registry = Mock()
        registry.log_pyfunc_model.return_value = "runs:/run-123/yolo_detector"
        registry.register_model.return_value = "12"
        registry.get_model_version_by_alias.return_value = Mock(tags={"map50": "0.6"})
        tracker = Mock()
        use_case = DetectionMlflowModelRegistry(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(
                    model_name="yolo_detector",
                    checkpoint_dir="data/checkpoint/yolo/weights",
                    best_checkpoint_name="best.pt",
                ),
            ),
            registry=registry,
            tracker=tracker,
            logger=Mock(),
        )

        with patch.object(
            use_case,
            "_load_checkpoint",
            return_value={"epoch": 1, "train_metrics": {"metrics/mAP50(B)": 0.6}},
        ):
            version = use_case.execute(run_id="run-123", pwd="/repo")

        self.assertEqual(version, "12")
        registry.set_model_version_tags.assert_called_once_with(
            registered_model_name="yolo_detector",
            model_version="12",
            tags={"map50": "0.6"},
        )
        registry.set_model_alias.assert_not_called()
        registry.transition_model_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
