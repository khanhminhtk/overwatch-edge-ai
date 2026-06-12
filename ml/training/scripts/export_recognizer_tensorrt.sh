#!/usr/bin/env bash
set -euo pipefail

MODE="local"

if [[ "${1:-}" == "--mode" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[error] --mode requires a value: local, dev, or prod." >&2
    exit 1
  fi

  MODE="$2"
  shift 2
fi

case "$MODE" in
  local)
    echo "[info] Running export with mode 'local'."
    echo "skipping export in 'local' mode since it may require GPU resources and specific environment setup."
    ;;

  dev|prod)
    echo "[info] Running export with mode '$MODE'."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${1:-artifacts/recognizer/best_loss.pt}"
ONNX_PATH="${2:-artifacts/onnx/recognizer.onnx}"
ENGINE_PATH="${3:-artifacts/tensorrt/recognizer.engine}"
PRECISION="${4:-fp16}"  # fp32 | fp16 | int8
CONFIG_PATH="${5:-config/training/recognizer_ctc.yaml}"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv_python/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python venv not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CONFIG_PATH

if ! command -v trtexec >/dev/null 2>&1; then
  echo "Error: trtexec not found. Please install TensorRT and ensure trtexec is in PATH." >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.infra.onnx.export_recognizer_onnx \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${ONNX_PATH}" \
  --config "${CONFIG_PATH}"

mkdir -p "$(dirname "${ENGINE_PATH}")"

read -r NUM_PATCHES PATCH_H PATCH_W <<< "$("${PYTHON_BIN}" - <<'PY'
import os
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader(
    yaml_relative_paths=[os.environ["CONFIG_PATH"]],
    env_relative_path="config/.env",
)

cfg = loader.load_recognizer().config_training["dataset"]

num_patches = int(cfg.get("num_patches", 16))
patch_size = cfg.get("patch_size", [64, 64])

patch_h = int(patch_size[0])
patch_w = int(patch_size[1])

print(num_patches, patch_h, patch_w)
PY
)"

TRT_FLAGS=()

case "${PRECISION}" in
  fp16)
    TRT_FLAGS+=(--fp16)
    ;;
  int8)
    TRT_FLAGS+=(--int8)
    ;;
  fp32)
    ;;
  *)
    echo "Error: unsupported precision '${PRECISION}'. Use fp32|fp16|int8." >&2
    exit 1
    ;;
esac

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --minShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --optShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --maxShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --skipInference \
  "${TRT_FLAGS[@]}"

echo "Recognizer TensorRT engine exported: ${ENGINE_PATH}"

    ;;

  *)
    echo "[error] Invalid mode: '$MODE'. Allowed values: local, dev, prod." >&2
    exit 1
    ;;
esac