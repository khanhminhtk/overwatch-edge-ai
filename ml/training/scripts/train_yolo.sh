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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

case "$MODE" in
  local)
    echo "[info] Skipping training recognizer orchestration with mode 'local'."
    ;;

  dev|prod)
    echo "[info] Running training recognizer orchestration with mode '$MODE'."
    uv run python -m src.application.use_cases.orchestration.train_recognizer_orchestration "$@"
    ;;

  *)
    echo "[error] Invalid mode: '$MODE'. Allowed values: local, dev, prod." >&2
    exit 1
    ;;
esac