#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WEIGHTS_PATH="${1:-artifacts/detection/yolo/weights/best.pt}"
OUTPUT_ENGINE_PATH="${2:-artifacts/tensorrt/yolo.engine}"
YOLO_CONFIG_PATH="${3:-config/training/yolo/config.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WEIGHTS_PATH OUTPUT_ENGINE_PATH YOLO_CONFIG_PATH

"${PROJECT_ROOT}/.venv/bin/python" - <<'PY'
from pathlib import Path
import os

from src.infra.modeling.detection.yolo import YoloTrainer
from src.utils.config_loader import ConfigLoader

weights_path = os.environ["WEIGHTS_PATH"]
output_engine_path = os.environ["OUTPUT_ENGINE_PATH"]
yolo_config_path = os.environ["YOLO_CONFIG_PATH"]

loader = ConfigLoader(
    yaml_relative_paths=[yolo_config_path],
    env_relative_path="config/.env",
)
config = loader.load_yolo_config(yolo_config_path)
trainer = YoloTrainer(config=config)
trainer.load_model_best_weights(weights_path)

output_path = Path(output_engine_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
trainer.export_tensorrt(output_path)
print(f"YOLO TensorRT engine exported: {output_path}")
PY
