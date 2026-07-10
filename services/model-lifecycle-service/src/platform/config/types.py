from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

PathLike = str | Path


class ConfigSource(Protocol):
    def load(self) -> dict[str, Any]: ...


class EnvSource(Protocol):
    def load(self) -> dict[str, str]: ...
