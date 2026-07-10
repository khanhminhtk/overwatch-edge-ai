from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class HardwareInfo:
    gpu_host_name: str = "unknown_host"
    gpu_name: str = "unknown_gpu"
    gpu_memory_gb: int = 0
    cuda_version: str = "unknown_cuda_version"
    driver_version: str = "unknown_driver_version"

    def __post_init__(self) -> None:
        if self.gpu_memory_gb < 0:
            raise ValueError("gpu_memory_gb must be non-negative")
