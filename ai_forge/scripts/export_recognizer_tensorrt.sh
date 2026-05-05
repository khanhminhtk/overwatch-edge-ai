#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${1:-artifacts/recognizer/best_loss.pt}"
ONNX_PATH="${2:-artifacts/onnx/recognizer.onnx}"
ENGINE_PATH="${3:-artifacts/tensorrt/recognizer.engine}"
PRECISION="${4:-fp16}"  # fp32 | fp16 | int8
CONFIG_PATH="${5:-config/training/recognizer_ctc.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CONFIG_PATH

if ! command -v trtexec >/dev/null 2>&1; then
  echo "Error: trtexec not found. Please install TensorRT and ensure trtexec is in PATH." >&2
  exit 1
fi

uv run -m src.infra.onnx.export_recognizer_onnx \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${ONNX_PATH}" \
  --config "${CONFIG_PATH}"

mkdir -p "$(dirname "${ENGINE_PATH}")"

read -r NUM_PATCHES PATCH_H PATCH_W <<< "$(uv run python - <<'PY'
from src.utils.config_loader import ConfigLoader
loader = ConfigLoader(
    yaml_relative_paths=[__import__("os").environ["CONFIG_PATH"]],
    env_relative_path="config/.env",
)
cfg = loader.load_recognizer().config_training["dataset"]
num_patches = int(cfg.get("num_patches", 16))
patch_h = int(cfg.get("patch_size", [64, 64])[0])
patch_w = int(cfg.get("patch_size", [64, 64])[1])
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
