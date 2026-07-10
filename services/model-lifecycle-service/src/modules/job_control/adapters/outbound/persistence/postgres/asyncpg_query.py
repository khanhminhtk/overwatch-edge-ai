from __future__ import annotations

import re
from typing import Any


_PARAM_PATTERN = re.compile(r"%\((?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\)s")


def bind_query(query: str, params: dict[str, Any]) -> tuple[str, list[Any]]:
    ordered_args: list[Any] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in params:
            raise KeyError(f"Missing SQL param: {name}")
        ordered_args.append(params[name])
        return f"${len(ordered_args)}"

    bound_query = _PARAM_PATTERN.sub(replace, query)
    return bound_query, ordered_args
