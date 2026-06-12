from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main_export_worker


class MainExportWorkerTest(unittest.TestCase):
    def test_main_builds_runtime_and_runs_worker_loop(self) -> None:
        worker = Mock()
        runtime = SimpleNamespace(worker=worker)

        with patch.object(main_export_worker, "create_worker_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_export_worker.Logger, "configure"):
                main_export_worker.main(
                    [
                        "--server-id",
                        "edge-03",
                        "--event-topic",
                        "export.events",
                        "--training-mode",
                        "docker",
                        "--idle-sleep-seconds",
                        "1.5",
                    ]
                )

        create_runtime_mock.assert_called_once()
        self.assertEqual(create_runtime_mock.call_args.kwargs["server_id"], "edge-03")
        self.assertEqual(create_runtime_mock.call_args.kwargs["event_topic"], "export.events")
        self.assertEqual(create_runtime_mock.call_args.kwargs["training_mode"], "docker")
        self.assertEqual(create_runtime_mock.call_args.kwargs["idle_sleep_seconds"], 1.5)
        worker.run_forever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
