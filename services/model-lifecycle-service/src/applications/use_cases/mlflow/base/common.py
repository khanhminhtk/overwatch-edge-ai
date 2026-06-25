from __future__ import annotations

import os

from src.utils.configloader import load_environment
from src.utils.logger import Logger


class BaseMlflowUseCase:
    def __init__(self, logger: Logger, path_env_ml_traning: str):
        self._logger = logger
        load_environment(path_env_ml_traning, override=True)

    @staticmethod
    def _env(key: str, default: str = "") -> str:
        return os.getenv(key, default)

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        return int(os.getenv(key, str(default)))

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        return float(os.getenv(key, str(default)))

    @staticmethod
    def _resolve_path(pwd: str | None, *parts: str) -> str:
        resolved_parts = [part for part in ((pwd or ""), *parts) if part]
        if not resolved_parts:
            return ""
        return os.path.join(*resolved_parts)
