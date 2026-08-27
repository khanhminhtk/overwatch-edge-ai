from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.exports.application.use_case import ExportRecognizerUseCase
from src.modules.exports.domain.value_objects import ExportSpec
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.platform.config import ConfigLoader
from src.platform.logger import Logger

REPO_ROOT = SERVICE_ROOT.parent.parent
CONFIG_YAML = REPO_ROOT / "services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"
CONFIG_ENV = REPO_ROOT / "services/model-lifecycle-service/config/.env"


def build_use_case() -> ExportRecognizerUseCase:
    recognizer_config = ConfigLoader.load(
        RecognizerConfig,
        yaml_files=[str(CONFIG_YAML)],
        env_files=[str(CONFIG_ENV)],
        section={
            "mlflow": None,
            "mlflow.recognizer": None,
        },
    )
    return ExportRecognizerUseCase(
        recognizer_config=recognizer_config,
        logger=Logger("ExportRecognizerUseCase"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export recognizer (ViT-CTC) .pt checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: <checkpoint_dir>/<best_checkpoint_name> from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/onnx/recognizer.onnx",
        help="Output ONNX path (default: artifacts/onnx/recognizer.onnx)",
    )
    parser.add_argument(
        "--exporter",
        type=str,
        choices=("legacy", "dynamo", "auto"),
        default="legacy",
        help="ONNX exporter path (default: legacy)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root for resolving relative paths (default: REPO_ROOT)",
    )
    return parser.parse_args()


def main() -> None:
    use_case = build_use_case()
    args = _parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else REPO_ROOT

    recognizer_config = use_case._recognizer_config
    if args.checkpoint is None:
        checkpoint_path = (
            Path(recognizer_config.checkpoint_dir)
            / recognizer_config.best_checkpoint_name
        )
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    spec = ExportSpec(
        model_type="recognizer",
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        exporter=args.exporter,
        project_root=project_root,
    )

    result = use_case.execute(spec)
    print(f"Recognizer ONNX exported: {result.output_path}")
    if result.duration_ms is not None:
        print(f"Duration: {result.duration_ms:.2f}ms")


if __name__ == "__main__":
    main()
