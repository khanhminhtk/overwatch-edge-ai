"""Domain ports for model inference."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DetectorModel(Protocol):
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
