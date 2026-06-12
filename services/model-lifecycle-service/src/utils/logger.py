from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TextIO


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass(frozen=True)
class LoggerConfig:
    level: int | str = logging.INFO
    fmt: str = DEFAULT_LOG_FORMAT
    stream: TextIO = field(default_factory=lambda: sys.stdout)


class Logger:
    def __init__(self, name: str | None = None):
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._logger.name

    @classmethod
    def configure(cls, config: LoggerConfig | None = None) -> None:
        resolved_config = config or LoggerConfig()
        root_logger = logging.getLogger()

        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

        handler = logging.StreamHandler(resolved_config.stream)
        handler.setFormatter(logging.Formatter(resolved_config.fmt))

        root_logger.addHandler(handler)
        root_logger.setLevel(cls._resolve_log_level(resolved_config.level))

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self._logger.debug(self._compose_message(message, *args), **kwargs)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self._logger.info(self._compose_message(message, *args), **kwargs)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self._logger.warning(self._compose_message(message, *args), **kwargs)

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self._logger.error(self._compose_message(message, *args), **kwargs)

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        self._logger.exception(self._compose_message(message, *args), **kwargs)

    @staticmethod
    def _compose_message(message: str, *args: object) -> str:
        if not args:
            return message
        return " ".join([message, *(str(arg) for arg in args)])

    @staticmethod
    def _resolve_log_level(level: int | str) -> int:
        if isinstance(level, int):
            return level

        resolved_level = getattr(logging, level.upper(), None)
        if isinstance(resolved_level, int):
            return resolved_level

        raise ValueError(f"Unsupported log level: {level}")
