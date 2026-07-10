from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.platform.tracking.mlflow.trace import MlflowTrace


class MlflowTraceUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.mock_client.tracking_uri = "http://mock-tracking:5000"

        patcher_client_mlflow = patch("src.platform.tracking.mlflow.client.mlflow")
        self.mock_mlflow = patcher_client_mlflow.start()
        self.addCleanup(patcher_client_mlflow.stop)

        patcher_trace_mlflow = patch(
            "src.platform.tracking.mlflow.trace.mlflow",
            self.mock_mlflow,
        )
        patcher_trace_mlflow.start()
        self.addCleanup(patcher_trace_mlflow.stop)

        self.trace = MlflowTrace(client=self.mock_client)

    def test_init_sets_tracking_uri_globally(self) -> None:
        self.mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://mock-tracking:5000"
        )

    def test_log_trace(self) -> None:
        self.mock_mlflow.log_trace.return_value = "trace-abc"

        result = self.trace.log_trace(
            name="inference",
            request={"input": "data"},
            response={"output": "result"},
            intermediate_outputs={"step1": "done"},
            attributes={"model": "v1"},
            tags={"env": "prod"},
            start_time_ms=1000,
            execution_time_ms=50,
        )

        self.assertEqual(result, "trace-abc")
        self.mock_mlflow.log_trace.assert_called_once_with(
            name="inference",
            request={"input": "data"},
            response={"output": "result"},
            intermediate_outputs={"step1": "done"},
            attributes={"model": "v1"},
            tags={"env": "prod"},
            start_time_ms=1000,
            execution_time_ms=50,
        )

    def test_log_trace_with_minimal_args(self) -> None:
        self.mock_mlflow.log_trace.return_value = "trace-min"

        result = self.trace.log_trace(name="minimal")

        self.assertEqual(result, "trace-min")
        self.mock_mlflow.log_trace.assert_called_once_with(
            name="minimal",
            request=None,
            response=None,
            intermediate_outputs=None,
            attributes=None,
            tags=None,
            start_time_ms=None,
            execution_time_ms=None,
        )

    def test_log_trace_metric(self) -> None:
        self.trace.log_trace_metric("run-id", "latency", 0.045)
        self.mock_client.log_metric.assert_called_once_with(
            run_id="run-id", key="latency", value=0.045
        )

    def test_log_trace_metric_converts_to_float(self) -> None:
        self.trace.log_trace_metric("run-id", "score", "0.9")
        self.mock_client.log_metric.assert_called_once_with(
            run_id="run-id", key="score", value=0.9
        )


if __name__ == "__main__":
    unittest.main()
