from __future__ import annotations


class JobControlError(Exception):
    """Base error for the job control domain."""


class DuplicateJobError(JobControlError):
    """Raised when a job already exists for the same business identity."""
