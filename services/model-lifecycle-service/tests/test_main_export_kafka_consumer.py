from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main_export_kafka_consumer


class MainExportKafkaConsumerTest(unittest.TestCase):
    def test_main_builds_runtime_and_runs_consumer_loop(self) -> None:
        handler = SimpleNamespace(handle_message=Mock())
        consumer = Mock()
        shutdown = Mock()
        runtime = SimpleNamespace(
            handler=handler,
            consumer=consumer,
            shutdown=shutdown,
        )

        with patch.object(main_export_kafka_consumer, "create_consumer_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_export_kafka_consumer.Logger, "configure"):
                main_export_kafka_consumer.main(
                    [
                        "--job-key",
                        "export_recognizer_tensorrt",
                        "--server-id",
                        "edge-02",
                        "--dlq-topic",
                        "export.dlq",
                    ]
                )

        create_runtime_mock.assert_called_once()
        self.assertEqual(create_runtime_mock.call_args.kwargs["job_key"], "export_recognizer_tensorrt")
        self.assertEqual(create_runtime_mock.call_args.kwargs["server_id"], "edge-02")
        self.assertEqual(create_runtime_mock.call_args.kwargs["dlq_topic"], "export.dlq")
        consumer.run.assert_called_once_with(handler.handle_message, shutdown)

    def test_main_passes_from_beginning_flag(self) -> None:
        runtime = SimpleNamespace(
            handler=SimpleNamespace(handle_message=Mock()),
            consumer=Mock(),
            shutdown=Mock(),
        )

        with patch.object(main_export_kafka_consumer, "create_consumer_runtime", return_value=runtime) as create_runtime_mock:
            with patch.object(main_export_kafka_consumer.Logger, "configure"):
                main_export_kafka_consumer.main(
                    [
                        "--job-key",
                        "export_recognizer_tensorrt",
                        "--from-beginning",
                    ]
                )

        self.assertTrue(create_runtime_mock.call_args.kwargs["from_beginning"])


if __name__ == "__main__":
    unittest.main()
