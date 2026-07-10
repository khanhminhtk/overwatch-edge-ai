from __future__ import annotations

from src.modules.job_control.domain.errors.duplicate_job_error import JobControlError


class InvalidJobStateError(JobControlError):
    """Raised when a job status transition violates the domain rules."""
