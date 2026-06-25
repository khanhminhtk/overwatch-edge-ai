from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


class ConfigProviderPort(ABC):
    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_typed_config(
        self,
        dto_cls: type[T],
        config_section: str | None = None,
    ) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_env_variable(self, var_name: str) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def env_files(self) -> list[str | Path]:
        raise NotImplementedError

    @property
    @abstractmethod
    def yaml_files(self) -> list[str | Path]:
        raise NotImplementedError
