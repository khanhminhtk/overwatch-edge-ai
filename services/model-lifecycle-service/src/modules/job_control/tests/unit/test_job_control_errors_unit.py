from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.domain.errors.duplicate_job_error import (  # noqa: E402
    DuplicateJobError,
    JobControlError,
)
from src.modules.job_control.domain.errors.invalid_event_error import (  # noqa: E402
    InvalidJobEventError,
)
from src.modules.job_control.domain.errors.invalid_job_state_error import (  # noqa: E402
    InvalidJobStateError,
)


class JobControlErrorsUnitTest(unittest.TestCase):
    def test_duplicate_job_error_inherits_job_control_error(self) -> None:
        error = DuplicateJobError("duplicate request_id=req-1")

        self.assertIsInstance(error, JobControlError)
        self.assertEqual(str(error), "duplicate request_id=req-1")

    def test_invalid_job_event_error_inherits_job_control_error(self) -> None:
        error = InvalidJobEventError("unsupported event_type=train_requested")

        self.assertIsInstance(error, JobControlError)
        self.assertEqual(str(error), "unsupported event_type=train_requested")

    def test_invalid_job_state_error_inherits_job_control_error(self) -> None:
        error = InvalidJobStateError("cannot mark processed from status=RECEIVED")

        self.assertIsInstance(error, JobControlError)
        self.assertEqual(str(error), "cannot mark processed from status=RECEIVED")


if __name__ == "__main__":
    unittest.main()
