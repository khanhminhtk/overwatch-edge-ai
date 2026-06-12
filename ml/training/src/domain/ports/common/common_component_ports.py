from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActivationPort(ABC):
    """Domain contract for activation-like components."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Apply activation transformation."""


class CacheableActivationPort(ActivationPort):
    """Activation contract with optional cache lifecycle."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear internal cache state."""


class PositionalEncodingPort(ABC):
    """Domain contract for positional encoding generators."""

    @abstractmethod
    def forward(
        self,
        seq_len: int,
        device: Any | None = None,
        has_cls_token: bool = False,
    ) -> tuple[Any, Any]:
        """Generate positional encoding tensors."""
