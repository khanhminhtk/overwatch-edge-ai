from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.exports.domain.value_objects import ExportResult, ExportSpec
from src.modules.tracking.application.use_case.tracking_common import log_exceptions
from src.modules.tracking.domain.value_objects import DetectionConfig
from src.platform.logger import Logger


class ExportDetectionUseCase:
    def __init__(
        self,
        detection_config: DetectionConfig,
        logger: Logger,
    ) -> None:
        self._detection_config = detection_config
        self._logger = logger

    @log_exceptions("[EXPORT_DETECTION_ERROR]")
    def execute(self, spec: ExportSpec) -> ExportResult:
        started_at = time.perf_counter()
        project_root = spec.project_root or Path.cwd()
        training_root = project_root / "ml/training"

        training_config_path = str(
            spec.training_config_path
            or Path(self._detection_config.training_config_path)
        )
        training_env_path = str(
            spec.training_env_path
            or Path(self._detection_config.training_env_path)
        )

        script = f"""
from pathlib import Path
import os, sys
sys.path.insert(0, '{training_root}')
from src.infra.modeling.detection.yolo import YoloTrainer
from src.utils.config_loader import ConfigLoader

project_root = Path('{project_root}')
config_path = str(project_root / '{training_config_path}')
env_path = str(project_root / '{training_env_path}')
checkpoint_path = '{spec.checkpoint_path}'
output_path = '{spec.output_path}'

loader = ConfigLoader(
    yaml_relative_paths=[config_path],
    project_root=project_root,
    env_relative_path=env_path,
)
config = loader.load_yolo_config(config_path)
trainer = YoloTrainer(config=config)
trainer.load_model_best_weights(checkpoint_path)
trainer.export_onnx(output_path)
print(f'Detection ONNX exported: {{output_path}}')
"""
        result = subprocess.run(
            ["uv", "run", "--no-sync", "--project", str(project_root), "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Detection export failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
            )

        duration_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info(
            "[EXPORT_DETECTION_SUCCEEDED]",
            f"output={spec.output_path}",
            f"duration_ms={duration_ms:.2f}",
        )
        return ExportResult(
            output_path=spec.output_path,
            success=True,
            duration_ms=duration_ms,
        )
