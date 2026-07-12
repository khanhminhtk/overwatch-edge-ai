from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.training.application.use_case.train_detection import TrainDetectionUseCase
from src.modules.training.domain.value_objects.training_spec import TrainingSpec
from src.platform.logger import Logger


class TrainDetectionUseCaseUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = MagicMock(spec=Logger)
        self.use_case = TrainDetectionUseCase(logger=self.logger)

    def test_execute_runs_script_with_correct_args(self) -> None:
        spec = TrainingSpec(
            model_name="detector",
            dataset_version="v2",
            mode="prod",
        )
        mock_process = MagicMock()
        mock_process.stdout = ["line1\n", "line2\n"]
        mock_process.returncode = 0
        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            result = self.use_case.execute(spec)

        self.assertTrue(result.success)
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args.args[0]
        self.assertIn("train_yolo.sh", str(cmd[1]))
        self.assertIn("--mode", cmd)
        self.assertIn("prod", cmd)
        self.assertIn("--dataset_version", cmd)
        self.assertIn("v2", cmd)

    def test_execute_raises_on_nonzero_exit(self) -> None:
        spec = TrainingSpec(
            model_name="detector",
            dataset_version="v2",
            mode="local",
        )
        mock_process = MagicMock()
        mock_process.stdout = ["error log\n"]
        mock_process.returncode = 1
        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            with self.assertRaises(RuntimeError) as ctx:
                self.use_case.execute(spec)
            self.assertIn("Detection training failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
