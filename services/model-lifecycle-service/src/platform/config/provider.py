from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from .mapper import map_to_dataclass, resolve_placeholders
from .sources import deep_merge
from .types import ConfigSource, EnvSource

T = TypeVar("T")


@dataclass
class ConfigProvider:
    config_sources: list[ConfigSource] = field(default_factory=list)
    env_sources: list[EnvSource] = field(default_factory=list)

    def get_raw(self, section: str | None = None) -> dict:
        merged = self._load_config()
        if section is None:
            return merged
        selected = self._select_section(merged, section)
        if not isinstance(selected, dict):
            raise ValueError(f"Config section must be a mapping: {section}")
        return selected

    def require(self, dto_cls: type[T], section: str | None = None) -> T:
        env = self._load_env()
        raw = self.get_raw(section=section)
        resolved = resolve_placeholders(raw, env)
        return map_to_dataclass(resolved, dto_cls)

    def _load_config(self) -> dict:
        merged: dict = {}
        for source in self.config_sources:
            loaded = source.load()
            if not isinstance(loaded, dict):
                raise ValueError(f"Config source must return a mapping: {source!r}")
            merged = deep_merge(merged, loaded)
        return merged

    def _load_env(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for source in self.env_sources:
            loaded = source.load()
            merged.update({key: value for key, value in loaded.items() if value is not None})
        return merged

    def _select_section(self, data: dict, section: str):
        current = data
        for part in section.split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Missing config section: {section}")
            current = current[part]
        return current
