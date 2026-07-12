from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.training.domain.value_objects.training_spec import TrainingResult, TrainingSpec
from src.platform.logger import Logger


class TrainDetectionUseCase:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def execute(self, spec: TrainingSpec) -> TrainingResult:
        started_at = time.perf_counter()
        project_root = SERVICE_ROOT.parent
        script_path = project_root / "ml" / "training" / "scripts" / "train_yolo.sh"

        cmd = ["bash", str(script_path), "--mode", spec.mode]
        if spec.dataset_version:
            cmd.extend(["--dataset_version", spec.dataset_version])
        if spec.train_config_uri:
            cmd.extend(["--train_config_uri", spec.train_config_uri])

        self._logger.info(
            "[TRAIN_DETECTION_STARTED]",
            f"script={script_path}",
            f"mode={spec.mode}",
            f"dataset_version={spec.dataset_version}",
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self._logger.info("[TRAIN_DETECTION_OUT]", line.rstrip())
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"Detection training failed with exit code {process.returncode}"
            )

        duration_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info(
            "[TRAIN_DETECTION_SUCCEEDED]",
            f"duration_ms={duration_ms:.2f}",
        )
        return TrainingResult(success=True, duration_ms=duration_ms)
