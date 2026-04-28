from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackBonePort(ABC):
    """Domain port for recognizer backbones."""

    @abstractmethod
    def freeze_feature_extractor(self) -> None:
        """Freeze feature extractor parameters."""

    @abstractmethod
    def unfreeze_feature_extractor(
        self,
        classifier: bool,
        num_layers_from_classification: int = 0,
    ) -> None:
        """Unfreeze part of feature extractor."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Map image patches to embeddings."""
