#!/usr/bin/env bash
set -euo pipefail

MODE="local"
PROJECT_ROOT="${PWD}"
ENV_PATH="config/.env"
CONFIG_PATH="ml/training/config/training/recognizer_ctc.yaml"
CHECKPOINT_PATH=""
ONNX_PATH="artifacts/onnx/recognizer.onnx"
ENGINE_PATH="artifacts/tensorrt/recognizer.engine"
PRECISION="fp16"

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
    --config-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --config-path requires a value." >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --checkpoint-path requires a value." >&2
        exit 1
      fi
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --onnx-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --onnx-path requires a value." >&2
        exit 1
      fi
      ONNX_PATH="$2"
      shift 2
      ;;
    --engine-path)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --engine-path requires a value." >&2
        exit 1
      fi
      ENGINE_PATH="$2"
      shift 2
      ;;
    --precision)
      if [[ -z "${2:-}" ]]; then
        echo "[error] --precision requires a value: fp32, fp16, or int8." >&2
        exit 1
      fi
      PRECISION="$2"
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
CONFIG_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${CONFIG_PATH}")"
ONNX_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${ONNX_PATH}")"
ENGINE_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${ENGINE_PATH}")"
UV_RUN_PYTHON=(uv run --project "${PROJECT_ROOT}" python)

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

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "[error] Config file not found: ${CONFIG_ABS_PATH}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv not found. Please install uv and ensure it is in PATH." >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "[error] trtexec not found. Please install TensorRT and ensure trtexec is in PATH." >&2
  exit 1
fi

export PROJECT_ROOT ENV_ABS_PATH CONFIG_ABS_PATH

mapfile -t DOTENV_EXPORTS < <(
  "${UV_RUN_PYTHON[@]}" - <<'PY'
import os
import shlex

from dotenv import dotenv_values

env_path = os.environ["ENV_ABS_PATH"]

for key, value in dotenv_values(env_path).items():
    if value is None:
        value = ""
    print(f"export {key}={shlex.quote(value)}")
PY
)

for export_cmd in "${DOTENV_EXPORTS[@]}"; do
  eval "${export_cmd}"
done

export PYTHONPATH="${TRAINING_ROOT}:${PYTHONPATH:-}"

mapfile -t RESOLVED_CONFIG_VALUES < <(
  cd "${TRAINING_ROOT}" && "${UV_RUN_PYTHON[@]}" - <<'PY'
from pathlib import Path
import os

from src.utils.config_loader import ConfigLoader

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
config_path = Path(os.environ["CONFIG_ABS_PATH"]).resolve()
env_path = Path(os.environ["ENV_ABS_PATH"]).resolve()

loader = ConfigLoader(
    yaml_relative_paths=[str(config_path)],
    project_root=project_root,
    env_relative_path=str(env_path),
)
config_training = loader.load_recognizer().config_training
dataset_cfg = dict(config_training["dataset"])
io_cfg = dict(config_training["io"])

checkpoint_name = str(
    io_cfg.get("best_cer_checkpoint_name")
    or io_cfg.get("best_checkpoint_name")
    or "best_cer.pt"
)
checkpoint_path = Path(io_cfg["save_dir"]) / checkpoint_name

patch_size = dataset_cfg.get("patch_size", [64, 64])
print(checkpoint_path)
print(int(dataset_cfg.get("num_patches", 16)))
print(int(patch_size[0]))
print(int(patch_size[1]))
PY
)

DEFAULT_CHECKPOINT_PATH="${RESOLVED_CONFIG_VALUES[0]}"
NUM_PATCHES="${RESOLVED_CONFIG_VALUES[1]}"
PATCH_H="${RESOLVED_CONFIG_VALUES[2]}"
PATCH_W="${RESOLVED_CONFIG_VALUES[3]}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ABS_PATH="${DEFAULT_CHECKPOINT_PATH}"
else
  CHECKPOINT_ABS_PATH="$(resolve_path "${PROJECT_ROOT}" "${CHECKPOINT_PATH}")"
fi

if [[ ! -f "${CHECKPOINT_ABS_PATH}" ]]; then
  echo "[error] Checkpoint file not found: ${CHECKPOINT_ABS_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${ONNX_ABS_PATH}")" "$(dirname "${ENGINE_ABS_PATH}")"

cd "${TRAINING_ROOT}"
"${UV_RUN_PYTHON[@]}" -m src.infra.onnx.export_recognizer_onnx \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${CHECKPOINT_ABS_PATH}" \
  --output "${ONNX_ABS_PATH}" \
  --config "${CONFIG_ABS_PATH}" \
  --env "${ENV_ABS_PATH}" \
  --exporter legacy

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
    echo "[error] Unsupported precision '${PRECISION}'. Use fp32|fp16|int8." >&2
    exit 1
    ;;
esac

trtexec \
  --onnx="${ONNX_ABS_PATH}" \
  --saveEngine="${ENGINE_ABS_PATH}" \
  --minShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --optShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --maxShapes=images:1x${NUM_PATCHES}x3x${PATCH_H}x${PATCH_W} \
  --skipInference \
  "${TRT_FLAGS[@]}"

echo "Recognizer TensorRT engine exported: ${ENGINE_ABS_PATH}"
