from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar, overload

from .mapper import map_to_dataclass, resolve_placeholders
from .provider import ConfigProvider
from .sources import DotenvEnvSource, EnvironmentEnvSource, YamlConfigSource
from .types import ConfigSource, EnvSource, PathLike

T = TypeVar("T")
T2 = TypeVar("T2")


class ConfigLoader:
    @overload
    @staticmethod
    def load(
        dto_cls: type[T],
        yaml_files: Sequence[PathLike] | None = ...,
        env_files: Sequence[PathLike] | None = ...,
        env: Mapping[str, str] | None = ...,
        include_os_env: bool = ...,
        section: str | Sequence[str] | Mapping[str, str | None] | None = ...,
        config_sources: Sequence[ConfigSource] | None = ...,
        env_sources: Sequence[EnvSource] | None = ...,
    ) -> T: ...

    @overload
    @staticmethod
    def load(
        dto_cls: tuple[type[T], type[T2]],
        yaml_files: Sequence[PathLike] | None = ...,
        env_files: Sequence[PathLike] | None = ...,
        env: Mapping[str, str] | None = ...,
        include_os_env: bool = ...,
        section: str | Sequence[str] | None = ...,
        config_sources: Sequence[ConfigSource] | None = ...,
        env_sources: Sequence[EnvSource] | None = ...,
    ) -> tuple[T, T2]: ...

    @staticmethod
    def load(
        dto_cls,
        yaml_files=None,
        env_files=None,
        env=None,
        include_os_env=True,
        section=None,
        config_sources=None,
        env_sources=None,
    ):
        all_config_sources: list[ConfigSource] = []
        all_env_sources: list[EnvSource] = []

        if config_sources:
            all_config_sources.extend(config_sources)
        elif yaml_files:
            all_config_sources.append(YamlConfigSource(list(yaml_files)))

        if env_sources:
            all_env_sources.extend(env_sources)
        else:
            if env_files:
                all_env_sources.append(DotenvEnvSource(list(env_files)))
            if include_os_env:
                all_env_sources.append(EnvironmentEnvSource())
            if env is not None:
                all_env_sources.append(EnvironmentEnvSource(env))

        provider = ConfigProvider(
            config_sources=all_config_sources,
            env_sources=all_env_sources,
        )

        if isinstance(dto_cls, tuple):
            if isinstance(section, list):
                if len(section) != len(dto_cls):
                    raise ValueError(
                        f"section length ({len(section)}) must match "
                        f"number of DTOs ({len(dto_cls)})"
                    )
                return tuple(
                    provider.require(cls, section=sec)
                    for cls, sec in zip(dto_cls, section)
                )
            return tuple(provider.require(cls, section=section) for cls in dto_cls)

        if isinstance(section, dict):
            merged: dict = {}
            for yaml_path, field_name in section.items():
                raw = provider.get_raw(section=yaml_path)
                if not isinstance(raw, dict):
                    raise ValueError(f"Config section must be a mapping: {yaml_path}")
                if field_name is None:
                    merged.update(raw)
                else:
                    merged[field_name] = raw
            env_vars = provider._load_env()
            resolved = resolve_placeholders(merged, env_vars)
            return map_to_dataclass(resolved, dto_cls)

        if isinstance(section, list):
            merged_list: dict = {}
            for sec in section:
                raw = provider.get_raw(section=sec)
                if not isinstance(raw, dict):
                    raise ValueError(f"Config section must be a mapping: {sec}")
                merged_list.update(raw)
            env_vars = provider._load_env()
            resolved = resolve_placeholders(merged_list, env_vars)
            return map_to_dataclass(resolved, dto_cls)

        return provider.require(dto_cls, section=section)
