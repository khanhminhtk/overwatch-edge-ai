#!/usr/bin/env bash
set -euo pipefail

MODE="local"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_PATH="ml/training/config/.env.example"
WEIGHTS_PATH="data/checkpoint/yolo/weights/best.pt"
OUTPUT_ENGINE_PATH="artifacts/onnx/detection.onnx"
YOLO_CONFIG_PATH="ml/training/config/training/yolo/config.yaml"

resolve_path() {
  local base_dir="$1"
  local path_value="$2"

  if [[ "${path_value}" = /* ]]; then
    printf '%s\n' "${path_value}"
    return
  fi

  printf '%s/%s\n' "${base_dir}" "${path_value}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --mode requires a value: local, dev, or prod." >&2
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    --project-root)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --project-root requires a value." >&2
        exit 1
      fi
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --env-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --env-path requires a value." >&2
        exit 1
      fi
      ENV_PATH="$2"
      shift 2
      ;;
    --weights-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --weights-path requires a value." >&2
        exit 1
      fi
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    --output-engine-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --output-engine-path requires a value." >&2
        exit 1
      fi
      OUTPUT_ENGINE_PATH="$2"
      shift 2
      ;;
    --yolo-config-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --yolo-config-path requires a value." >&2
        exit 1
      fi
      YOLO_CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
TRAINING_ROOT="${PROJECT_ROOT}/ml/training"
ENV_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${ENV_PATH}")"
WEIGHTS_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${WEIGHTS_PATH}")"
OUTPUT_ENGINE_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${OUTPUT_ENGINE_PATH}")"
YOLO_CONFIG_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${YOLO_CONFIG_PATH}")"
UV_RUN_PYTHON=(uv run --no-sync --project "${PROJECT_ROOT}" python)

if [[ ! -d "${TRAINING_ROOT}/src" ]]; then
  echo "[error] Invalid project root: '${PROJECT_ROOT}'. Expected '${TRAINING_ROOT}/src' to exist." >&2
  exit 1
fi

case "${MODE}" in
  local)
    echo "[info] Running export with mode 'local'."
    echo "skipping export in 'local' mode since it may require GPU resources and specific environment setup."
    exit 0
    ;;
  dev|prod)
    echo "[info] Running export with mode '${MODE}'."
    ;;
  *)
    echo "[error] Invalid mode: '${MODE}'. Allowed values: local, dev, prod." >&2
    exit 1
    ;;
esac

if [[ ! -f "${ENV_ABS_PATH}" ]]; then
  echo "[error] Env file not found: ${ENV_ABS_PATH}" >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS_ABS_PATH}" ]]; then
  echo "[error] Weights file not found: ${WEIGHTS_ABS_PATH}" >&2
  exit 1
fi

if [[ ! -f "${YOLO_CONFIG_ABS_PATH}" ]]; then
  echo "[error] YOLO config file not found: ${YOLO_CONFIG_ABS_PATH}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv not found. Please install uv and ensure it is in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ENGINE_ABS_PATH}")"

export PROJECT_ROOT ENV_ABS_PATH WEIGHTS_ABS_PATH OUTPUT_ENGINE_ABS_PATH YOLO_CONFIG_ABS_PATH

cd "${TRAINING_ROOT}"
PYTHONPATH="${TRAINING_ROOT}:${PYTHONPATH:-}" "${UV_RUN_PYTHON[@]}" - <<'PY'
from pathlib import Path
import os

from src.infra.modeling.detection.yolo import YoloTrainer
from src.utils.config_loader import ConfigLoader

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
env_path = Path(os.environ["ENV_ABS_PATH"]).resolve()
weights_path = Path(os.environ["WEIGHTS_ABS_PATH"]).resolve()
output_engine_path = Path(os.environ["OUTPUT_ENGINE_ABS_PATH"]).resolve()
yolo_config_path = Path(os.environ["YOLO_CONFIG_ABS_PATH"]).resolve()

loader = ConfigLoader(
    yaml_relative_paths=[str(yolo_config_path)],
    project_root=project_root,
    env_relative_path=str(env_path),
)
config = loader.load_yolo_config(str(yolo_config_path))
trainer = YoloTrainer(config=config)
trainer.load_model_best_weights(str(weights_path))
trainer.export_onnx(output_engine_path)
print(f"YOLO ONNX model exported: {output_engine_path}")
PY
