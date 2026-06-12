from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main_kafka_consumer


class MainKafkaConsumerTest(unittest.TestCase):
    def test_main_builds_runtime_and_runs_consumer_loop(self) -> None:
        handler = SimpleNamespace(handle_message=Mock())
        consumer = Mock()
        shutdown = Mock()
        runtime = SimpleNamespace(
            handler=handler,
            consumer=consumer,
            shutdown=shutdown,
        )

        with patch.object(main_kafka_consumer, "create_consumer_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_kafka_consumer.Logger, "configure"):
                main_kafka_consumer.main(
                    [
                        "--job-key",
                        "train_detection",
                        "--server-id",
                        "edge-02",
                        "--dlq-topic",
                        "training.dlq",
                    ]
                )

        create_runtime_mock.assert_called_once()
        self.assertEqual(create_runtime_mock.call_args.kwargs["job_keys"], ["train_detection"])
        self.assertEqual(create_runtime_mock.call_args.kwargs["server_id"], "edge-02")
        self.assertEqual(create_runtime_mock.call_args.kwargs["dlq_topic"], "training.dlq")
        consumer.run.assert_called_once_with(handler.handle_message, shutdown)

    def test_main_passes_from_beginning_flag(self) -> None:
        runtime = SimpleNamespace(
            handler=SimpleNamespace(handle_message=Mock()),
            consumer=Mock(),
            shutdown=Mock(),
        )

        with patch.object(main_kafka_consumer, "create_consumer_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_kafka_consumer.Logger, "configure"):
                main_kafka_consumer.main(
                    [
                        "--job-key",
                        "train_recognizer",
                        "--from-beginning",
                    ]
                )

        self.assertTrue(create_runtime_mock.call_args.kwargs["from_beginning"])

    def test_main_accepts_multiple_job_keys(self) -> None:
        runtime = SimpleNamespace(
            handler=SimpleNamespace(handle_message=Mock()),
            consumer=Mock(),
            shutdown=Mock(),
        )

        with patch.object(main_kafka_consumer, "create_consumer_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_kafka_consumer.Logger, "configure"):
                main_kafka_consumer.main(
                    [
                        "--job-key",
                        "train_recognizer",
                        "--job-key",
                        "train_detection",
                    ]
                )

        self.assertEqual(
            create_runtime_mock.call_args.kwargs["job_keys"],
            ["train_recognizer", "train_detection"],
        )


if __name__ == "__main__":
    unittest.main()
