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
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.platform.logger import Logger


class ExportRecognizerUseCase:
    def __init__(
        self,
        recognizer_config: RecognizerConfig,
        logger: Logger,
    ) -> None:
        self._recognizer_config = recognizer_config
        self._logger = logger

    @log_exceptions("[EXPORT_RECOGNIZER_ERROR]")
    def execute(self, spec: ExportSpec) -> ExportResult:
        started_at = time.perf_counter()
        project_root = spec.project_root or Path.cwd()
        training_root = project_root / "ml/training"

        training_config_path = str(
            spec.training_config_path
            or Path(self._recognizer_config.training_config_path)
        )
        training_env_path = str(
            spec.training_env_path
            or Path(self._recognizer_config.training_env_path)
        )

        script = f"""
import sys
sys.path.insert(0, '{training_root}')
from src.infra.onnx.export_recognizer_onnx import export_recognizer_to_onnx

out = export_recognizer_to_onnx(
    checkpoint_path='{spec.checkpoint_path}',
    output_path='{spec.output_path}',
    config_relative_path='{training_config_path}',
    env_relative_path='{training_env_path}',
    project_root='{project_root}',
    exporter='{spec.exporter}',
)
print(f'Recognizer ONNX exported: {{out}}')
"""
        result = subprocess.run(
            ["uv", "run", "--no-sync", "--project", str(project_root), "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Recognizer export failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
            )

        duration_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info(
            "[EXPORT_RECOGNIZER_SUCCEEDED]",
            f"output={spec.output_path}",
            f"duration_ms={duration_ms:.2f}",
        )
        return ExportResult(
            output_path=spec.output_path,
            success=True,
            duration_ms=duration_ms,
        )
