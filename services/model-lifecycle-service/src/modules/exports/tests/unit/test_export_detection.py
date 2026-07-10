from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.exports.application.use_case.export_detection import (
    ExportDetectionUseCase,
)
from src.modules.exports.domain.value_objects import ExportResult, ExportSpec
from src.modules.tracking.domain.value_objects import DetectionConfig
from src.platform.logger import Logger


class ExportDetectionUseCaseUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detection_config = DetectionConfig(
            checkpoint_dir="/tmp/checkpoints",
            best_checkpoint_name="best.pt",
            training_config_path="ml/training/config/training/yolo/config.yaml",
            training_env_path="ml/training/config/.env.example",
        )
        self.logger = MagicMock(spec=Logger)
        self.use_case = ExportDetectionUseCase(
            detection_config=self.detection_config,
            logger=self.logger,
        )

    @patch(
        "src.modules.exports.application.use_case.export_detection.subprocess.run",
        autospec=True,
    )
    def test_execute_calls_subprocess(
        self,
        mock_subprocess_run: MagicMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Detection ONNX exported: /tmp/onnx/detection.onnx\n"
        mock_subprocess_run.return_value = mock_result

        spec = ExportSpec(
            model_type="detection",
            checkpoint_path=Path("/tmp/checkpoints/best.pt"),
            output_path=Path("/tmp/onnx/detection.onnx"),
            project_root=Path("/project"),
        )

        result = self.use_case.execute(spec)

        self.assertTrue(result.success)
        self.assertEqual(result.output_path, Path("/tmp/onnx/detection.onnx"))
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        self.assertIn("uv", args[0])
        self.assertIn("python", args[0])
        self.assertIn("-c", args[0])
        self.assertEqual(kwargs["timeout"], 600)

    @patch(
        "src.modules.exports.application.use_case.export_detection.subprocess.run",
        autospec=True,
    )
    def test_execute_raises_on_failure(
        self,
        mock_subprocess_run: MagicMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: checkpoint not found"
        mock_subprocess_run.return_value = mock_result

        spec = ExportSpec(
            model_type="detection",
            checkpoint_path=Path("/tmp/checkpoints/missing.pt"),
            output_path=Path("/tmp/onnx/detection.onnx"),
            project_root=Path("/project"),
        )

        with self.assertRaisesRegex(RuntimeError, "Detection export failed"):
            self.use_case.execute(spec)

    def test_default_checkpoint_path_from_config(self) -> None:
        config = self.detection_config
        default_checkpoint = Path(config.checkpoint_dir) / config.best_checkpoint_name
        self.assertEqual(default_checkpoint, Path("/tmp/checkpoints/best.pt"))


if __name__ == "__main__":
    unittest.main()
