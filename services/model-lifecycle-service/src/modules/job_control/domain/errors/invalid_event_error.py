from __future__ import annotations

from src.modules.job_control.domain.errors.duplicate_job_error import JobControlError


class InvalidJobEventError(JobControlError):
    """Raised when an inbound event cannot be accepted into job control."""
