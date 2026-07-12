from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class JobReclaimConfig:
    default_processing_timeout_seconds: int
    overrides: dict[str, int] = field(default_factory=dict)

    def timeout_for(self, job_name: str) -> int:
        return self.overrides.get(job_name, self.default_processing_timeout_seconds)


@dataclass(frozen=True)
class JobControlConfig:
    result_topic: str
    reclaim: JobReclaimConfig
