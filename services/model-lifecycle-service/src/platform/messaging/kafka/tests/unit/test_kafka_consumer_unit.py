from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, PropertyMock, call, patch

from confluent_kafka import KafkaError, KafkaException

from src.platform.messaging.kafka.consumer import KafkaConsumerClient
from src.platform.messaging.kafka.message import (
    ConsumedMessage,
    KafkaMessageProcessingError,
)


class KafkaConsumerClientInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.bootstrap_servers = "broker:9092"
        self.config.security_protocol = "PLAINTEXT"
        self.config.client_id_prefix = "test-svc"
        self.config.consumer_config.auto_offset_reset = "earliest"
        self.config.consumer_config.enable_auto_commit = False
        self.config.consumer_config.session_timeout_ms = 45000
        self.config.consumer_config.max_poll_interval_ms = 300000

        self.logger = MagicMock()
        self.handler = MagicMock()

    def test_init_creates_underlying_consumer(self) -> None:
        client = KafkaConsumerClient(
            config=self.config,
            handler=self.handler,
            group_id="test-group",
            topics=["topic-a", "topic-b"],
            logger=self.logger,
        )
        self.assertEqual(client._group_id, "test-group")
        self.assertEqual(client._topics, ["topic-a", "topic-b"])
        self.assertIsInstance(client._consumer, type(client).__bases__[0])
        self.assertFalse(client._started)
        self.assertFalse(client._closed)
        self.assertFalse(client._stop_event.is_set())

    def test_empty_topics_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one topic"):
            KafkaConsumerClient(
                config=self.config,
                handler=self.handler,
                group_id="g",
                topics=[],
                logger=self.logger,
            )

    def test_default_poll_timeout(self) -> None:
        client = KafkaConsumerClient(
            config=self.config,
            handler=self.handler,
            group_id="g",
            topics=["t"],
            logger=self.logger,
        )
        self.assertEqual(client._poll_timeout, 1.0)

    def test_custom_poll_timeout(self) -> None:
        client = KafkaConsumerClient(
            config=self.config,
            handler=self.handler,
            group_id="g",
            topics=["t"],
            logger=self.logger,
            poll_timeout=5.0,
        )
        self.assertEqual(client._poll_timeout, 5.0)


class KafkaConsumerClientRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.bootstrap_servers = "broker:9092"
        self.config.security_protocol = "PLAINTEXT"
        self.config.client_id_prefix = "test-svc"
        self.config.consumer_config.auto_offset_reset = "earliest"
        self.config.consumer_config.enable_auto_commit = False
        self.config.consumer_config.session_timeout_ms = 45000
        self.config.consumer_config.max_poll_interval_ms = 300000

        self.logger = MagicMock()
        self.handler = MagicMock()

        self.client = KafkaConsumerClient(
            config=self.config,
            handler=self.handler,
            group_id="test-group",
            topics=["t"],
            logger=self.logger,
            poll_timeout=0.01,
        )
        # Replace real consumer backend with a mock
        self.mock_consumer = MagicMock()
        self.client._consumer = self.mock_consumer

    def _run_with_poll_results(self, *results: object) -> None:
        """Run client.run(), feeding poll results one by one.
        After the last explicit result the stop event is set so the loop exits.
        """
        poll_results = list(results)

        def _poll(timeout: float | None = None) -> object:
            if not poll_results:
                self.client._stop_event.set()
                return None
            result = poll_results.pop(0)
            if not poll_results:
                self.client._stop_event.set()
            return result

        self.mock_consumer.poll.side_effect = _poll
        self.client.run()

    def _make_message(
        self,
        topic: str = "t",
        partition: int = 0,
        offset: int = 0,
        key: bytes | None = None,
        value: bytes = b"v",
        headers: object = None,
        timestamp: tuple[int, int] = (0, 0),
        error: object = None,
    ) -> MagicMock:
        msg = MagicMock()
        msg.error.return_value = error
        msg.topic.return_value = topic
        msg.partition.return_value = partition
        msg.offset.return_value = offset
        msg.key.return_value = key
        msg.value.return_value = value
        msg.headers.return_value = headers
        msg.timestamp.return_value = timestamp
        return msg

    # --- tests that need the poll loop to execute ---

    def test_run_subscribes_to_topics(self) -> None:
        self._run_with_poll_results(None)
        self.mock_consumer.subscribe.assert_called_once_with(["t"])

    def test_run_polls_with_timeout(self) -> None:
        self._run_with_poll_results(None)
        self.mock_consumer.poll.assert_called_once_with(timeout=0.01)

    def test_run_sets_started_flag(self) -> None:
        self._run_with_poll_results(None)
        self.assertTrue(self.client._started)

    def test_run_raises_if_already_started(self) -> None:
        self.client._started = True
        with self.assertRaises(RuntimeError):
            self.client.run()

    def test_run_raises_if_closed(self) -> None:
        self.client._closed = True
        with self.assertRaises(RuntimeError):
            self.client.run()

    def test_run_processes_message_via_handler(self) -> None:
        mock_msg = self._make_message(
            topic="t",
            partition=0,
            offset=42,
            key=b"k",
            value=b"v",
            headers=(("h", b"hv"),),
            timestamp=(1, 1712345678000),
        )

        self._run_with_poll_results(mock_msg)

        self.handler.assert_called_once()
        consumed: ConsumedMessage = self.handler.call_args[0][0]
        self.assertIsInstance(consumed, ConsumedMessage)
        self.assertEqual(consumed.topic, "t")
        self.assertEqual(consumed.partition, 0)
        self.assertEqual(consumed.offset, 42)
        self.assertEqual(consumed.key, b"k")
        self.assertEqual(consumed.value, b"v")
        self.assertEqual(consumed.headers, (("h", b"hv"),))
        self.assertEqual(consumed.timestamp_ms, 1712345678000)
        self.assertEqual(consumed.timestamp_type, 1)

    def test_run_commits_after_successful_processing(self) -> None:
        mock_msg = self._make_message(offset=1)

        self._run_with_poll_results(mock_msg)

        self.mock_consumer.commit.assert_called_once_with(
            message=mock_msg, asynchronous=False
        )

    def test_run_skips_poll_timeout(self) -> None:
        self._run_with_poll_results(None, None)

        self.assertEqual(self.mock_consumer.poll.call_count, 2)
        self.handler.assert_not_called()
        self.mock_consumer.commit.assert_not_called()

    def test_run_skips_partition_eof(self) -> None:
        mock_msg = self._make_message(
            error=MagicMock(code=lambda: KafkaError._PARTITION_EOF),
        )

        self._run_with_poll_results(mock_msg)

        self.handler.assert_not_called()
        self.mock_consumer.commit.assert_not_called()

    def test_run_raises_on_consumer_kafka_error(self) -> None:
        mock_msg = self._make_message(
            error=MagicMock(code=lambda: KafkaError.BROKER_NOT_AVAILABLE),
        )

        self.mock_consumer.poll.return_value = mock_msg

        with self.assertRaises(KafkaException):
            self.client.run()

    def test_run_raises_kafka_message_processing_error_on_handler_failure(
        self,
    ) -> None:
        self.handler.side_effect = ValueError("handler failed")

        mock_msg = self._make_message(offset=99)

        self.mock_consumer.poll.return_value = mock_msg

        with self.assertRaises(KafkaMessageProcessingError) as cm:
            self.client.run()

        self.assertEqual(cm.exception.message.offset, 99)

    def test_run_closes_consumer_in_finally_on_success(self) -> None:
        self._run_with_poll_results(None)
        self.mock_consumer.close.assert_called_once()
        self.assertTrue(self.client._closed)

    def test_run_closes_consumer_in_finally_on_error(self) -> None:
        self.handler.side_effect = ValueError()

        mock_msg = self._make_message()

        self.mock_consumer.poll.return_value = mock_msg

        with self.assertRaises(KafkaMessageProcessingError):
            self.client.run()

        self.mock_consumer.close.assert_called_once()
        self.assertTrue(self.client._closed)


class KafkaConsumerClientStopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.bootstrap_servers = "broker:9092"
        self.config.security_protocol = "PLAINTEXT"
        self.config.client_id_prefix = "test-svc"
        self.config.consumer_config.auto_offset_reset = "earliest"
        self.config.consumer_config.enable_auto_commit = False
        self.config.consumer_config.session_timeout_ms = 45000
        self.config.consumer_config.max_poll_interval_ms = 300000

        self.logger = MagicMock()
        self.handler = MagicMock()

        self.client = KafkaConsumerClient(
            config=self.config,
            handler=self.handler,
            group_id="g",
            topics=["t"],
            logger=self.logger,
        )

    def test_stop_sets_stop_event(self) -> None:
        self.assertFalse(self.client._stop_event.is_set())
        self.client.stop()
        self.assertTrue(self.client._stop_event.is_set())

    def test_stop_is_idempotent(self) -> None:
        self.client.stop()
        self.client.stop()
        self.assertTrue(self.client._stop_event.is_set())
        self.assertEqual(self.logger.info.call_count, 1)

    def test_stop_logs_event(self) -> None:
        self.client.stop()
        self.logger.info.assert_called_once()
        log_arg = self.logger.info.call_args[0][0]
        self.assertIn("STOP_REQUESTED", log_arg)

    def test_stop_causes_run_to_exit(self) -> None:
        # Run in a thread so we can call stop() concurrently
        self.client._stop_event.clear()
        self.client._consumer = MagicMock()
        self.client._consumer.poll.return_value = None

        result: list[bool] = []

        def target() -> None:
            self.client.run()
            result.append(True)

        t = threading.Thread(target=target, daemon=True)
        t.start()

        self.client.stop()
        t.join(timeout=2)
        self.assertEqual(result, [True])


class KafkaConsumerClientHasErrorTest(unittest.TestCase):
    def test_no_error_returns_false(self) -> None:
        msg = MagicMock()
        msg.error.return_value = None
        self.assertFalse(KafkaConsumerClient._has_error(msg))

    def test_partition_eof_returns_true(self) -> None:
        msg = MagicMock()
        err = MagicMock()
        err.code.return_value = KafkaError._PARTITION_EOF
        msg.error.return_value = err
        self.assertTrue(KafkaConsumerClient._has_error(msg))

    def test_other_error_raises_kafka_exception(self) -> None:
        msg = MagicMock()
        err = MagicMock()
        err.code.return_value = KafkaError.BROKER_NOT_AVAILABLE
        msg.error.return_value = err
        with self.assertRaises(KafkaException):
            KafkaConsumerClient._has_error(msg)


class KafkaConsumerClientToConfluentConfigTest(unittest.TestCase):
    def test_returns_correct_dict(self) -> None:
        config = MagicMock()
        config.bootstrap_servers = "host:9092"
        config.security_protocol = "SSL"
        config.client_id_prefix = "my-svc"
        config.consumer_config.auto_offset_reset = "latest"
        config.consumer_config.enable_auto_commit = True
        config.consumer_config.session_timeout_ms = 30000
        config.consumer_config.max_poll_interval_ms = 600000

        result = KafkaConsumerClient._to_confluent_config(
            config, "my-group"
        )
        expected = {
            "bootstrap.servers": "host:9092",
            "security.protocol": "SSL",
            "group.id": "my-group",
            "client.id": "my-svc",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 600000,
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
