from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactPort(Protocol):
    def execute(
        self,
        checkpoint_name: list[str],
        run_id: str | None = None,
        pwd: str | None = None,
    ) -> None: ...
