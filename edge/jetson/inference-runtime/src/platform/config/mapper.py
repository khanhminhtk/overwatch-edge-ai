from __future__ import annotations

import re
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def resolve_placeholders(data: Any, env: dict[str, str], path: str = "") -> Any:
    if isinstance(data, dict):
        return {
            key: resolve_placeholders(value, env, f"{path}.{key}" if path else str(key))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            resolve_placeholders(item, env, f"{path}[{index}]")
            for index, item in enumerate(data)
        ]
    if isinstance(data, tuple):
        return tuple(
            resolve_placeholders(item, env, f"{path}[{index}]")
            for index, item in enumerate(data)
        )
    if isinstance(data, str):
        return _resolve_string(data, env, path)
    return data


def _resolve_string(value: str, env: dict[str, str], path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(3)
        if key in env:
            return env[key]
        if default is not None:
            return default
        raise ValueError(f"Missing environment variable: {key} at {path or '<root>'}")

    return ENV_PATTERN.sub(replace, value)


def map_to_dataclass(data: Any, target_type: type[T] | Any, path: str = "") -> T:
    if not is_dataclass(target_type):
        return cast_value(data, target_type, path)
    if not isinstance(data, dict):
        raise _invalid_type_error(path, target_type, data)

    mapped_values: dict[str, Any] = {}
    type_hints = get_type_hints(target_type)
    for field in fields(target_type):
        field_path = f"{path}.{field.name}" if path else field.name
        field_type = type_hints.get(field.name, field.type)
        if field.name not in data:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise ValueError(f"Missing config field: {field_path}")
        mapped_values[field.name] = cast_value(data[field.name], field_type, field_path)
    return target_type(**mapped_values)


def cast_value(value: Any, target_type: Any, path: str = "") -> Any:
    if target_type is Any:
        return value

    if _is_optional(target_type):
        if value is None:
            return None
        non_none = next(arg for arg in get_args(target_type) if arg is not type(None))
        return cast_value(value, non_none, path)

    if _is_union(target_type):
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

    if is_dataclass(target_type):
        return map_to_dataclass(value, target_type, path)

    origin = get_origin(target_type)
    if origin is list:
        if not isinstance(value, list):
            raise _invalid_type_error(path, target_type, value)
        item_type = get_args(target_type)[0] if get_args(target_type) else Any
        return [
            cast_value(item, item_type, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin is dict:
        if not isinstance(value, dict):
            raise _invalid_type_error(path, target_type, value)
        key_type, value_type = get_args(target_type) or (Any, Any)
        return {
            cast_value(
                item_key, key_type, f"{path}.<key>" if path else "<key>"
            ): cast_value(
                item_value,
                value_type,
                f"{path}.{item_key}" if path else str(item_key),
            )
            for item_key, item_value in value.items()
        }

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise _invalid_type_error(path, target_type, value)
        item_types = get_args(target_type)
        items = list(value)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(
                cast_value(item, item_types[0], f"{path}[{index}]")
                for index, item in enumerate(items)
            )
        if len(item_types) != len(items):
            raise _invalid_type_error(path, target_type, value)
        return tuple(
            cast_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(items, item_types))
        )

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        if isinstance(value, target_type):
            return value
        if isinstance(value, str):
            try:
                return target_type(value)
            except ValueError as exc:
                raise _invalid_type_error(path, target_type, value) from exc
        raise _invalid_type_error(path, target_type, value)

    if target_type is Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise _invalid_type_error(path, target_type, value)

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

    if isinstance(target_type, type):
        if isinstance(value, target_type):
            return value
        raise _invalid_type_error(path, target_type, value)

    return value


def _is_optional(target_type: Any) -> bool:
    origin = get_origin(target_type)
    return origin in (UnionType, Union) and type(None) in get_args(target_type)


def _is_union(target_type: Any) -> bool:
    origin = get_origin(target_type)
    return origin in (UnionType, Union)


def _expected_type_name(target_type: Any) -> str:
    if target_type is Any:
        return "Any"
    if target_type is Path:
        return "Path"
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type.__name__
    origin = get_origin(target_type)
    if origin is None:
        return getattr(target_type, "__name__", str(target_type))
    if origin is list:
        args = get_args(target_type)
        return f"list[{_expected_type_name(args[0])}]" if args else "list"
    if origin is dict:
        key_type, value_type = get_args(target_type) or (Any, Any)
        return (
            f"dict[{_expected_type_name(key_type)}, {_expected_type_name(value_type)}]"
        )
    if origin is tuple:
        args = get_args(target_type)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_expected_type_name(args[0])}, ...]"
        return f"tuple[{', '.join(_expected_type_name(arg) for arg in args)}]"
    if _is_union(target_type):
        return " | ".join(_expected_type_name(arg) for arg in get_args(target_type))
    return str(target_type)


def _invalid_type_error(path: str, target_type: Any, value: Any) -> ValueError:
    location = path or "<root>"
    if isinstance(value, str):
        return ValueError(
            f'Invalid config type at {location}: expected {_expected_type_name(target_type)}, got "{value}"'
        )
    return ValueError(
        f"Invalid config type at {location}: expected {_expected_type_name(target_type)}, got {value!r}"
    )
