from __future__ import annotations

import os
import re
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml
from dotenv import load_dotenv


T = TypeVar("T")
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _format_path(path: str) -> str:
    return path or "<root>"


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _is_optional_type(target_type: Any) -> bool:
    origin = get_origin(target_type)
    if origin not in (UnionType, Union):
        return False
    return type(None) in get_args(target_type)


def _is_union_type(target_type: Any) -> bool:
    origin = get_origin(target_type)
    return origin in (UnionType, Union)


def _strip_optional_type(target_type: Any) -> Any:
    return next(arg for arg in get_args(target_type) if arg is not type(None))


def _expected_type_name(target_type: Any) -> str:
    if target_type is Any:
        return "Any"
    if target_type is None or target_type is type(None):
        return "None"
    origin = get_origin(target_type)
    if origin is None:
        return getattr(target_type, "__name__", str(target_type))
    if origin is list:
        args = get_args(target_type)
        return f"list[{_expected_type_name(args[0])}]" if args else "list"
    if origin is dict:
        key_type, value_type = get_args(target_type) or (Any, Any)
        return f"dict[{_expected_type_name(key_type)}, {_expected_type_name(value_type)}]"
    if origin is tuple:
        args = get_args(target_type)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_expected_type_name(args[0])}, ...]"
        return f"tuple[{', '.join(_expected_type_name(arg) for arg in args)}]"
    if _is_union_type(target_type):
        return " | ".join(_expected_type_name(arg) for arg in get_args(target_type))
    return str(target_type)


def _invalid_type_error(path: str, target_type: Any, value: Any) -> ValueError:
    return ValueError(
        f'Invalid config type at {_format_path(path)}: expected {_expected_type_name(target_type)}, got {value!r}'
        if not isinstance(value, str)
        else f'Invalid config type at {_format_path(path)}: expected {_expected_type_name(target_type)}, got "{value}"'
    )


def load_yaml_config(config_path: str | Path) -> Any:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_environment(env_file: str | Path | None = None) -> None:
    if env_file is None:
        return
    load_dotenv(dotenv_path=env_file, override=False)


def resolve_env_placeholders(data: Any, path: str = "") -> Any:
    if isinstance(data, dict):
        return {
            key: resolve_env_placeholders(
                value,
                f"{path}.{key}" if path else str(key),
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            resolve_env_placeholders(item, f"{path}[{index}]")
            for index, item in enumerate(data)
        ]

    if isinstance(data, tuple):
        return tuple(
            resolve_env_placeholders(item, f"{path}[{index}]")
            for index, item in enumerate(data)
        )

    if isinstance(data, str):
        return _resolve_env_string(data, path)

    return data


def _resolve_env_string(value: str, path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        default_value = match.group(3)
        env_var_value = os.getenv(env_var_name)
        if env_var_value is not None:
            return env_var_value
        if default_value is not None:
            return default_value
        raise ValueError(
            f"Missing environment variable: {env_var_name} at {_format_path(path)}"
        )

    return ENV_PATTERN.sub(replace, value)


def cast_value(value: Any, target_type: Any, path: str = "") -> Any:
    if target_type is Any:
        return value

    if _is_optional_type(target_type):
        if value is None:
            return None
        return cast_value(value, _strip_optional_type(target_type), path)

    if _is_union_type(target_type):
        last_error: ValueError | None = None
        for option in get_args(target_type):
            if option is type(None):
                continue
            try:
                return cast_value(value, option, path)
            except ValueError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return value

    if value is None:
        raise _invalid_type_error(path, target_type, value)

    origin = get_origin(target_type)
    if is_dataclass(target_type):
        return map_to_dataclass(value, target_type, path)

    if origin is dict:
        if not isinstance(value, dict):
            raise _invalid_type_error(path, target_type, value)
        key_type, value_type = get_args(target_type) or (Any, Any)
        return {
            cast_value(item_key, key_type, f"{path}.<key>" if path else "<key>"): cast_value(
                item_value,
                value_type,
                f"{path}.{item_key}" if path else str(item_key),
            )
            for item_key, item_value in value.items()
        }

    if origin is list:
        if not isinstance(value, list):
            raise _invalid_type_error(path, target_type, value)
        item_type = get_args(target_type)[0] if get_args(target_type) else Any
        return [
            cast_value(item, item_type, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise _invalid_type_error(path, target_type, value)
        args = get_args(target_type)
        items = list(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                cast_value(item, args[0], f"{path}[{index}]")
                for index, item in enumerate(items)
            )
        if len(args) != len(items):
            raise _invalid_type_error(path, target_type, value)
        return tuple(
            cast_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(items, args))
        )

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise _invalid_type_error(path, target_type, value)

    if target_type is int:
        if isinstance(value, bool):
            raise _invalid_type_error(path, target_type, value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise _invalid_type_error(path, target_type, value) from exc
        raise _invalid_type_error(path, target_type, value)

    if target_type is float:
        if isinstance(value, bool):
            raise _invalid_type_error(path, target_type, value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise _invalid_type_error(path, target_type, value) from exc
        raise _invalid_type_error(path, target_type, value)

    if target_type is str:
        if isinstance(value, str):
            return value
        raise _invalid_type_error(path, target_type, value)

    if target_type in (dict, list, tuple):
        return target_type(value)

    if isinstance(target_type, type):
        if isinstance(value, target_type):
            return value
        raise _invalid_type_error(path, target_type, value)

    return value


def map_to_dataclass(data: Any, target_type: type[T] | Any, path: str = "") -> T:
    if not is_dataclass(target_type):
        return cast_value(data, target_type, path)

    if not isinstance(data, dict):
        raise _invalid_type_error(path, target_type, data)

    type_hints = get_type_hints(target_type)
    mapped_values: dict[str, Any] = {}

    for field in fields(target_type):
        field_path = f"{path}.{field.name}" if path else field.name
        field_type = type_hints.get(field.name, field.type)

        if field.name not in data:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise ValueError(f"Missing config field: {field_path}")

        mapped_values[field.name] = cast_value(data[field.name], field_type, field_path)

    return target_type(**mapped_values)


def load_config(
    config_path: str | Path,
    dto_cls: type[T],
    env_file: str | Path | None = None,
    config_section: str | None = None,
) -> T:
    load_environment(env_file)
    raw_config = load_yaml_config(config_path)
    resolved_config = resolve_env_placeholders(raw_config)
    resolved_config = _select_config_section(
        resolved_config,
        dto_cls,
        config_section=config_section,
    )
    return map_to_dataclass(resolved_config, dto_cls)


def _select_config_section(
    data: Any,
    dto_cls: type[T],
    config_section: str | None = None,
) -> Any:
    if not is_dataclass(dto_cls) or not isinstance(data, dict):
        return data

    if config_section is not None:
        return _get_config_section(data, config_section)

    if any(field.name in data for field in fields(dto_cls)):
        return data

    class_name = dto_cls.__name__
    candidate_keys = []
    if class_name.endswith("Config"):
        candidate_keys.append(_camel_to_snake(class_name[:-6]))
    candidate_keys.append(_camel_to_snake(class_name))

    for candidate_key in candidate_keys:
        candidate_value = data.get(candidate_key)
        if isinstance(candidate_value, dict):
            return candidate_value

    return data


def _get_config_section(data: dict[str, Any], config_section: str) -> Any:
    current: Any = data
    for section_part in config_section.split("."):
        if not isinstance(current, dict) or section_part not in current:
            raise ValueError(f"Missing config section: {config_section}")
        current = current[section_part]
    return current


class ConfigLoader:
    def __init__(self, env_file: str | Path | None, config_yaml_file: str | Path):
        self.env_file = env_file
        self.config_yaml_file = config_yaml_file
        self.config: dict[str, Any] = {}
        self._env_loaded = False

    def _load_env_variables(self) -> None:
        if not self._env_loaded:
            load_environment(self.env_file)
            self._env_loaded = True

    def _load_config(self) -> dict[str, Any]:
        return load_yaml_config(self.config_yaml_file)

    def get_config(self) -> dict[str, Any]:
        self._load_env_variables()
        self.config = resolve_env_placeholders(self._load_config())
        return self.config

    def get_typed_config(
        self,
        dto_cls: type[T],
        config_section: str | None = None,
    ) -> T:
        self._load_env_variables()
        return load_config(
            config_path=self.config_yaml_file,
            dto_cls=dto_cls,
            env_file=self.env_file,
            config_section=config_section,
        )

    def get_env_variable(self, var_name: str) -> str | None:
        self._load_env_variables()
        return os.getenv(var_name)


# if __name__ == "__main__":
#     from src.infra.config.postgrest import PostgrestConfig
#     from src.infra.config.kafka import KafkaConfig
#     config_loader = ConfigLoader(
#         env_file="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/config/.env", 
#         config_yaml_file="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml")
    
#     config = config_loader.get_typed_config(PostgrestConfig, config_section="PostgresSql")
#     print(config)
#     print('================================')
#     kafka_config = config_loader.get_typed_config(KafkaConfig, config_section="kafka")
#     print(kafka_config)
