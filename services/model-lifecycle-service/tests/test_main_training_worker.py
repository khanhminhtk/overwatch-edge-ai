from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main_training_worker


class MainTrainingWorkerTest(unittest.TestCase):
    def test_main_builds_runtime_and_runs_worker_loop(self) -> None:
        worker = Mock()
        runtime = SimpleNamespace(worker=worker)

        with patch.object(main_training_worker, "create_worker_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_training_worker.Logger, "configure"):
                main_training_worker.main(
                    [
                        "--server-id",
                        "edge-03",
                        "--event-topic",
                        "training.events",
                        "--training-mode",
                        "local",
                        "--idle-sleep-seconds",
                        "1.5",
                    ]
                )

        create_runtime_mock.assert_called_once()
        self.assertEqual(create_runtime_mock.call_args.kwargs["server_id"], "edge-03")
        self.assertEqual(create_runtime_mock.call_args.kwargs["event_topic"], "training.events")
        self.assertEqual(create_runtime_mock.call_args.kwargs["training_mode"], "local")
        self.assertEqual(create_runtime_mock.call_args.kwargs["idle_sleep_seconds"], 1.5)
        worker.run_forever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
