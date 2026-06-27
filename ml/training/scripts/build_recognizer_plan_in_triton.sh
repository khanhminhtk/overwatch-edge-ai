#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PWD}"
ENV_FILE="config/.env"
MODEL_ENV_FILE="ml/training/config/.env.example"
COMPOSE_FILE="deploy/docker/compose/dockercompose.triton.yaml"
MODE="build"
MODEL_VERSION="${TRITON_MODEL_VERSION:-1}"
CONFIG_FILE="ml/training/config/training/recognizer_ctc.yaml"
ONNX_PATH="artifacts/onnx/recognizer.onnx"
CHECKPOINT_PATH="data/checkpoint_recognizer/best_loss.pt"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -n "${SUDO_USER:-}" && -x "/home/${SUDO_USER}/.local/bin/uv" ]]; then
  UV_BIN="/home/${SUDO_USER}/.local/bin/uv"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
else
  echo "[error] uv not found in PATH." >&2
  exit 1
fi

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
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --onnx-path)
      ONNX_PATH="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --version)
      MODEL_VERSION="$2"
      shift 2
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

(
  cd "${PROJECT_ROOT}/ml/training"
  PYTHONPATH="${PROJECT_ROOT}/ml/training:${PYTHONPATH:-}" \
    "${UV_BIN}" run --project "${PROJECT_ROOT}" python -m src.infra.onnx.export_recognizer_onnx \
      --project-root "${PROJECT_ROOT}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --output "${ONNX_PATH}" \
      --config "${CONFIG_FILE}" \
      --env "${MODEL_ENV_FILE}" \
      --exporter legacy
)

case "${MODE}" in
  build)
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile recognizer-builder build triton_builder_recognizer
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile recognizer-builder run --rm triton_builder_recognizer
    ;;
  debug)
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile recognizer-builder build triton_builder_recognizer
    TRITON_MODEL_VERSION="${MODEL_VERSION}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" --profile recognizer-builder run --rm triton_builder_recognizer bash
    ;;
  *)
    echo "[error] Unsupported mode '${MODE}'. Use build or debug." >&2
    exit 1
    ;;
esac
