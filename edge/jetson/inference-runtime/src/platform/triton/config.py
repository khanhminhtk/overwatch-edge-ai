from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TritonConfig:
    url: str
    model_name: str
    model_version: str = "1"
    verbose: bool = False
