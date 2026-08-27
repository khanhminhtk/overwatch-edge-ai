#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="config/.env"
MODEL_ENV_FILE="ml/training/config/.env.example"
COMPOSE_FILE="deploy/docker/compose/dockercompose.triton.yaml"
MODE="build"
MODEL_VERSION="${TRITON_MODEL_VERSION:-1}"
WEIGHTS_PATH="data/checkpoint/yolo/weights/best.pt"
ONNX_PATH="artifacts/onnx/detection.onnx"
YOLO_CONFIG_PATH="ml/training/config/training/yolo/config.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --model-env-file)
      MODEL_ENV_FILE="$2"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="$2"
      shift 2
      ;;
    --weights-path)
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    --onnx-path)
      ONNX_PATH="$2"
      shift 2
      ;;
    --yolo-config-path)
      YOLO_CONFIG_PATH="$2"
      shift 2
      ;;
    --version)
      MODEL_VERSION="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

bash ml/training/scripts/export_yolo_onnx_cpu.sh \
  --mode prod \
  --project-root "${PROJECT_ROOT}" \
  --env-path "${MODEL_ENV_FILE}" \
  --weights-path "${WEIGHTS_PATH}" \
  --output-engine-path "${ONNX_PATH}" \
  --yolo-config-path "${YOLO_CONFIG_PATH}"

case "${MODE}" in
  build)
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile detector-builder build triton_builder_detector
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile detector-builder run --rm triton_builder_detector
    ;;
  debug)
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile detector-builder build triton_builder_detector
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile detector-builder run --rm triton_builder_detector bash
    ;;
  *)
    echo "[error] Unsupported mode '${MODE}'. Use build or debug." >&2
    exit 1
    ;;
esac
