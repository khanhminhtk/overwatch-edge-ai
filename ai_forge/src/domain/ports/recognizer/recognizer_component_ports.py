from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AttentionPort(ABC):
    """Domain contract for attention-like recognizer components."""

    @abstractmethod
    def forward(
        self,
        x: Any,
        mask: Any | None = None,
        attn_mask: Any | None = None,
        has_cls_token: bool = False,
    ) -> Any:
        """Apply attention over patch/token sequences."""


class MoEPort(ABC):
    """Domain contract for Mixture-of-Experts recognizer components."""

    last_aux_loss: Any

    @abstractmethod
    def forward(self, x: Any) -> tuple[Any, Any | None]:
        """Return transformed features and optional auxiliary routing loss."""


class RecognizerBlockPort(ABC):
    """Generic domain contract for recognizer blocks."""

    @abstractmethod
    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform input data for the next recognizer stage."""
