from abc import ABC, abstractmethod
from typing import Any

class VisionTransformerPort(ABC):
    @abstractmethod
    def forward(
        self,
        x: Any,
        attn_mask: Any | None = None,
        has_cls_token: bool = False,
    ) -> Any:
        """Forward recognizer transformer block(s)."""
