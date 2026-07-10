from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from .types import PathLike


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


@dataclass(frozen=True)
class YamlConfigSource:
    paths: list[PathLike]

    def load(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for current_path in self.paths:
            try:
                with Path(current_path).open("r", encoding="utf-8") as file:
                    loaded = yaml.safe_load(file) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Failed to parse YAML file {current_path}: {exc}"
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"Root YAML config must be a mapping: {current_path}")
            merged = deep_merge(merged, loaded)
        return merged


@dataclass(frozen=True)
class DotenvEnvSource:
    paths: list[PathLike]

    def load(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for current_path in self.paths:
            values = dotenv_values(Path(current_path))
            for key, value in values.items():
                if value is None:
                    continue
                merged[key] = value
        return merged


@dataclass(frozen=True)
class EnvironmentEnvSource:
    values: Mapping[str, str] | None = None

    def load(self) -> dict[str, str]:
        if self.values is None:
            from os import environ

            return dict(environ)
        return dict(self.values)
