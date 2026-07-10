from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.exports.application.use_case.export_recognizer import (
    ExportRecognizerUseCase,
)
from src.modules.exports.domain.value_objects import ExportResult, ExportSpec
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.platform.logger import Logger


class ExportRecognizerUseCaseUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer_config = RecognizerConfig(
            checkpoint_dir="/tmp/checkpoints_recognizer",
            best_checkpoint_name="best_cer.pt",
            training_config_path="ml/training/config/training/recognizer_ctc.yaml",
            training_env_path="ml/training/config/.env.example",
        )
        self.logger = MagicMock(spec=Logger)
        self.use_case = ExportRecognizerUseCase(
            recognizer_config=self.recognizer_config,
            logger=self.logger,
        )

    @patch(
        "src.modules.exports.application.use_case.export_recognizer.subprocess.run",
        autospec=True,
    )
    def test_execute_calls_subprocess(
        self,
        mock_subprocess_run: MagicMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Recognizer ONNX exported: /tmp/onnx/recognizer.onnx\n"
        mock_subprocess_run.return_value = mock_result

        spec = ExportSpec(
            model_type="recognizer",
            checkpoint_path=Path("/tmp/checkpoints_recognizer/best_cer.pt"),
            output_path=Path("/tmp/onnx/recognizer.onnx"),
            exporter="legacy",
            project_root=Path("/project"),
        )

        result = self.use_case.execute(spec)

        self.assertTrue(result.success)
        self.assertEqual(result.output_path, Path("/tmp/onnx/recognizer.onnx"))
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        self.assertIn("uv", args[0])
        self.assertIn("python", args[0])
        self.assertIn("-c", args[0])
        self.assertEqual(kwargs["timeout"], 600)

    @patch(
        "src.modules.exports.application.use_case.export_recognizer.subprocess.run",
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
            model_type="recognizer",
            checkpoint_path=Path("/tmp/checkpoints_recognizer/missing.pt"),
            output_path=Path("/tmp/onnx/recognizer.onnx"),
            project_root=Path("/project"),
        )

        with self.assertRaisesRegex(RuntimeError, "Recognizer export failed"):
            self.use_case.execute(spec)

    def test_default_checkpoint_path_from_config(self) -> None:
        config = self.recognizer_config
        default_checkpoint = Path(config.checkpoint_dir) / config.best_checkpoint_name
        self.assertEqual(default_checkpoint, Path("/tmp/checkpoints_recognizer/best_cer.pt"))


if __name__ == "__main__":
    unittest.main()
