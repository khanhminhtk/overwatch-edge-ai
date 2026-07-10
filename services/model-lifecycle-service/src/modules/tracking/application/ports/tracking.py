from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TrackingPort(Protocol):
    def start_run(self) -> str: ...

    def execute(self, run_id: str, pwd: str) -> None: ...

    def end_run(self, run_id: str, status: str = "FINISHED") -> None: ...
