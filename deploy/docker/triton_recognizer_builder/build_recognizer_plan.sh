#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

MODEL_KIND="${TRITON_MODEL_KIND:-recognizer}"
MODEL_VERSION="${TRITON_MODEL_VERSION:-1}"
WEIGHTS_PATH="${TRITON_WEIGHTS_PATH:-/workspace/data/checkpoint_recognizer/best_cer.pt}"
MODEL_ENV_PATH="${TRITON_MODEL_ENV_PATH:-/workspace/ml/training/config/.env.example}"
RECOGNIZER_CONFIG_PATH="${TRITON_RECOGNIZER_CONFIG_PATH:-/workspace/ml/training/config/training/recognizer_ctc.yaml}"
EXPORTER="${TRITON_RECOGNIZER_EXPORTER:-legacy}"
ONNX_PATH="${TRITON_ONNX_PATH:-/workspace/artifacts/onnx/${MODEL_KIND}.onnx}"
PLAN_PATH="${TRITON_PLAN_PATH:-/workspace/edge/jetson/triton/${MODEL_KIND}/${MODEL_VERSION}/${MODEL_KIND}.plan}"
MIN_SHAPES="${TRITON_MIN_SHAPES:-images:1x16x3x64x64}"
OPT_SHAPES="${TRITON_OPT_SHAPES:-images:1x16x3x64x64}"
MAX_SHAPES="${TRITON_MAX_SHAPES:-images:8x16x3x64x64}"
PRECISION_FLAG="${TRITON_PRECISION_FLAG:---fp16}"
TRTEXEC_BIN="${TRTEXEC_BIN:-}"
EXTRA_FLAGS="${TRITON_TRTEXEC_EXTRA_FLAGS:-}"

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
  echo "[error] recognizer weights file not found: ${WEIGHTS_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${ONNX_PATH}")"
mkdir -p "$(dirname "${PLAN_PATH}")"

echo "[info] TensorRT builder image: ${NVIDIA_TRITON_SERVER_VERSION:-unknown}"
echo "[info] model kind: ${MODEL_KIND}"
echo "[info] model version: ${MODEL_VERSION}"
echo "[info] weights path: ${WEIGHTS_PATH}"
echo "[info] model env path: ${MODEL_ENV_PATH}"
echo "[info] recognizer config path: ${RECOGNIZER_CONFIG_PATH}"
echo "[info] ONNX path: ${ONNX_PATH}"
echo "[info] PLAN path: ${PLAN_PATH}"
echo "[info] min shapes: ${MIN_SHAPES}"
echo "[info] opt shapes: ${OPT_SHAPES}"
echo "[info] max shapes: ${MAX_SHAPES}"
echo "[info] precision flag: ${PRECISION_FLAG}"
echo "[info] extra flags: ${EXTRA_FLAGS}"

(
  cd /workspace/ml/training
  PYTHONPATH="/workspace/ml/training:${PYTHONPATH:-}" \
    uv run --no-sync --project /workspace python -m src.infra.onnx.export_recognizer_onnx \
      --project-root /workspace \
      --checkpoint "${WEIGHTS_PATH}" \
      --output "${ONNX_PATH}" \
      --config "${RECOGNIZER_CONFIG_PATH}" \
      --env "${MODEL_ENV_PATH}" \
      --exporter "${EXPORTER}"
)

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[error] ONNX file not found after recognizer export: ${ONNX_PATH}" >&2
  exit 1
fi

if [[ "${TRITON_ENV:-prod}" != "dev" ]]; then
  if [[ -z "${TRTEXEC_BIN}" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
      TRTEXEC_BIN="$(command -v trtexec)"
    elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
      TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
    else
      echo "[error] trtexec not found in PATH or /usr/src/tensorrt/bin/trtexec" >&2
      exit 1
    fi
  fi
  echo "[info] trtexec: ${TRTEXEC_BIN}"

  # shellcheck disable=SC2206
  extra_args=(${EXTRA_FLAGS})

  "${TRTEXEC_BIN}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${PLAN_PATH}" \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --skipInference \
    "${PRECISION_FLAG}" \
    "${extra_args[@]}"

  echo "[info] Engine build finished"
  sha256sum "${PLAN_PATH}"
fi
