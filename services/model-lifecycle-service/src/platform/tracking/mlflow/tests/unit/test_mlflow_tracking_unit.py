from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.tracking import MlflowTracking


class MlflowTrackingUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=MlflowClient)
        self.mock_client.tracking_uri = "http://mock-tracking:5000"
        self.tracking = MlflowTracking(client=self.mock_client)

    def test_init_stores_client(self) -> None:
        self.assertIs(self.tracking._client, self.mock_client)

    def test_ensure_experiment_returns_existing(self) -> None:
        mock_exp = MagicMock()
        mock_exp.experiment_id = "existing-id"
        self.mock_client.get_experiment_by_name.return_value = mock_exp

        result = self.tracking.ensure_experiment("my-experiment")

        self.assertEqual(result, "existing-id")
        self.mock_client.create_experiment.assert_not_called()

    def test_ensure_experiment_creates_new(self) -> None:
        self.mock_client.get_experiment_by_name.return_value = None
        self.mock_client.create_experiment.return_value = "new-id"

        result = self.tracking.ensure_experiment("new-experiment")

        self.assertEqual(result, "new-id")
        self.mock_client.create_experiment.assert_called_once_with(
            "new-experiment"
        )

    def test_start_run_creates_run(self) -> None:
        self.mock_client.get_experiment_by_name.return_value = None
        self.mock_client.create_experiment.return_value = "exp-id"
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        self.mock_client.create_run.return_value = mock_run

        result = self.tracking.start_run("exp", "my-run")

        self.assertEqual(result, "run-123")
        self.mock_client.create_run.assert_called_once_with(
            experiment_id="exp-id",
            run_name="my-run",
            tags={"mlflow.runName": "my-run"},
        )

    def test_resume_run_returns_run_id(self) -> None:
        result = self.tracking.resume_run("run-abc")
        self.assertEqual(result, "run-abc")
        self.mock_client.get_run.assert_called_once_with("run-abc")

    def test_end_run_sets_terminated(self) -> None:
        self.tracking.end_run("run-xyz", status="FINISHED")
        self.mock_client.set_terminated.assert_called_once_with(
            "run-xyz", status="FINISHED"
        )

    def test_end_run_default_status(self) -> None:
        self.tracking.end_run("run-xyz")
        self.mock_client.set_terminated.assert_called_once_with(
            "run-xyz", status="FINISHED"
        )

    def test_set_tags(self) -> None:
        self.tracking.set_tags("run-1", {"env": "prod", "ver": "2"})
        self.mock_client.set_tag.assert_any_call("run-1", "env", "prod")
        self.mock_client.set_tag.assert_any_call("run-1", "ver", "2")

    def test_log_params(self) -> None:
        self.tracking.log_params("run-1", {"lr": 0.01, "epochs": 10})
        self.mock_client.log_param.assert_any_call("run-1", "lr", "0.01")
        self.mock_client.log_param.assert_any_call(
            "run-1", "epochs", "10"
        )

    def test_log_metrics(self) -> None:
        self.tracking.log_metrics(
            "run-1", {"accuracy": 0.95, "loss": 0.05}
        )
        self.mock_client.log_metric.assert_any_call(
            "run-1", "accuracy", 0.95
        )
        self.mock_client.log_metric.assert_any_call("run-1", "loss", 0.05)

    def test_log_metric_series(self) -> None:
        rows = [
            {"value": 0.9, "step": 0, "timestamp_ms": 1000},
            {"value": 0.95, "step": 1, "timestamp_ms": 2000},
        ]
        self.tracking.log_metric_series("run-1", "acc", rows)
        self.assertEqual(self.mock_client.log_metric.call_count, 2)

    def test_log_metric_series_handles_missing_timestamp(self) -> None:
        rows = [{"value": 0.9, "step": 0}]
        self.tracking.log_metric_series("run-1", "acc", rows)
        self.mock_client.log_metric.assert_called_once_with(
            run_id="run-1",
            key="acc",
            value=0.9,
            step=0,
            timestamp=None,
        )

    def test_client_property(self) -> None:
        self.assertIs(self.tracking.client, self.mock_client)


if __name__ == "__main__":
    unittest.main()
