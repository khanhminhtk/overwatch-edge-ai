from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RegistryPort(Protocol):
    def execute(self, run_id: str, pwd: str) -> str: ...
