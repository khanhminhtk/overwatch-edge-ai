from __future__ import annotations

import os
import unittest

from src.platform.tracking.mlflow.tracking import MlflowTracking

_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")


@unittest.skipIf(
    _MLFLOW_URI is None,
    "MLFLOW_TRACKING_URI environment variable not set",
)
class MlflowTrackingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert _MLFLOW_URI is not None
        cls.tracking = MlflowTracking(tracking_url=_MLFLOW_URI)

    def test_ensure_experiment_creates_and_returns_id(self) -> None:
        exp_name = "test-integration-experiment"
        exp_id = self.tracking.ensure_experiment(exp_name)
        self.assertIsInstance(exp_id, str)
        self.assertTrue(len(exp_id) > 0)

    def test_ensure_experiment_is_idempotent(self) -> None:
        exp_name = "test-integration-experiment-idempotent"
        id1 = self.tracking.ensure_experiment(exp_name)
        id2 = self.tracking.ensure_experiment(exp_name)
        self.assertEqual(id1, id2)

    def test_start_and_end_run(self) -> None:
        run_id = self.tracking.start_run(
            "test-integration-experiment", "integration-run"
        )
        self.assertIsInstance(run_id, str)
        self.assertTrue(len(run_id) > 0)
        self.tracking.end_run(run_id, status="FINISHED")

    def test_resume_run(self) -> None:
        run_id = self.tracking.start_run(
            "test-integration-experiment", "resume-test"
        )
        resumed = self.tracking.resume_run(run_id)
        self.assertEqual(resumed, run_id)
        self.tracking.end_run(run_id, status="FINISHED")


if __name__ == "__main__":
    unittest.main()
