from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from src.platform.messaging.kafka.message import (
    ConsumedMessage,
    KafkaMessageProcessingError,
)


class ConsumedMessageValidationTest(unittest.TestCase):
    def make_valid(self, **overrides) -> ConsumedMessage:
        kwargs = dict(
            topic="test-topic",
            partition=0,
            offset=100,
            key=b"test-key",
            value=b"test-value",
            headers=(("h1", b"v1"), ("h2", None)),
            timestamp_ms=1712345678000,
            timestamp_type=1,
            leader_epoch=0,
        )
        kwargs.update(overrides)
        return ConsumedMessage(**kwargs)

    def test_creates_with_valid_fields(self) -> None:
        msg = self.make_valid()
        self.assertEqual(msg.topic, "test-topic")
        self.assertEqual(msg.partition, 0)
        self.assertEqual(msg.offset, 100)
        self.assertEqual(msg.key, b"test-key")
        self.assertEqual(msg.value, b"test-value")
        self.assertEqual(msg.headers, (("h1", b"v1"), ("h2", None)))
        self.assertEqual(msg.timestamp_ms, 1712345678000)
        self.assertEqual(msg.timestamp_type, 1)
        self.assertEqual(msg.leader_epoch, 0)

    def test_whitespace_only_topic_raises(self) -> None:
        for topic in ("", "  ", "\t"):
            with self.subTest(topic=repr(topic)):
                with self.assertRaises(ValueError):
                    self.make_valid(topic=topic)

    def test_negative_partition_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.make_valid(partition=-1)

    def test_zero_partition_is_valid(self) -> None:
        msg = self.make_valid(partition=0)
        self.assertEqual(msg.partition, 0)

    def test_negative_offset_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.make_valid(offset=-1)

    def test_zero_offset_is_valid(self) -> None:
        msg = self.make_valid(offset=0)
        self.assertEqual(msg.offset, 0)

    def test_negative_timestamp_ms_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.make_valid(timestamp_ms=-1)

    def test_none_timestamp_ms_is_valid(self) -> None:
        msg = self.make_valid(timestamp_ms=None)
        self.assertIsNone(msg.timestamp_ms)

    def test_minimal_construction(self) -> None:
        msg = ConsumedMessage(
            topic="t",
            partition=0,
            offset=0,
            key=None,
            value=b"",
            headers=(),
        )
        self.assertEqual(msg.topic, "t")
        self.assertEqual(msg.value, b"")
        self.assertIsNone(msg.key)
        self.assertEqual(msg.headers, ())
        self.assertIsNone(msg.timestamp_ms)
        self.assertIsNone(msg.timestamp_type)
        self.assertIsNone(msg.leader_epoch)

    def test_is_frozen(self) -> None:
        msg = self.make_valid()
        with self.assertRaises(FrozenInstanceError):
            msg.topic = "other"

    def test_two_instances_equal_when_fields_match(self) -> None:
        kwargs = dict(
            topic="t",
            partition=0,
            offset=0,
            key=None,
            value=b"",
            headers=(),
        )
        self.assertEqual(
            ConsumedMessage(**kwargs), ConsumedMessage(**kwargs)
        )


class ConsumedMessagePropertyTest(unittest.TestCase):
    def make_valid(self, **overrides) -> ConsumedMessage:
        kwargs = dict(
            topic="test-topic",
            partition=0,
            offset=100,
            key=b"test-key",
            value=b"test-value",
        )
        kwargs.update(overrides)
        return ConsumedMessage(**kwargs)

    def test_identity(self) -> None:
        msg = self.make_valid(topic="t", partition=1, offset=42)
        self.assertEqual(msg.identity, ("t", 1, 42))

    def test_next_offset(self) -> None:
        msg = self.make_valid(offset=99)
        self.assertEqual(msg.next_offset, 100)

    def test_value_size_bytes_with_bytes(self) -> None:
        msg = self.make_valid(value=b"hello")
        self.assertEqual(msg.value_size_bytes, 5)

    def test_value_size_bytes_with_empty(self) -> None:
        msg = self.make_valid(value=b"")
        self.assertEqual(msg.value_size_bytes, 0)

    def test_timestamp_with_ms(self) -> None:
        msg = self.make_valid(timestamp_ms=1712345678000)
        expected = datetime.fromtimestamp(1712345678, tz=timezone.utc)
        self.assertEqual(msg.timestamp, expected)

    def test_timestamp_none_when_ms_none(self) -> None:
        msg = self.make_valid(timestamp_ms=None)
        self.assertIsNone(msg.timestamp)


class ConsumedMessageMethodTest(unittest.TestCase):
    def make_valid(self, **overrides) -> ConsumedMessage:
        kwargs = dict(
            topic="test-topic",
            partition=0,
            offset=100,
            key=b"test-key",
            value=b"test-value",
        )
        kwargs.update(overrides)
        return ConsumedMessage(**kwargs)

    def test_require_value_returns_bytes(self) -> None:
        msg = self.make_valid(value=b"data")
        self.assertEqual(msg.require_value(), b"data")

    def test_header_values_returns_all_matching(self) -> None:
        msg = self.make_valid(
            headers=(("h1", b"a"), ("h2", b"b"), ("h1", b"c")),
        )
        self.assertEqual(msg.header_values("h1"), (b"a", b"c"))
        self.assertEqual(msg.header_values("h2"), (b"b",))
        self.assertEqual(msg.header_values("nonexistent"), ())

    def test_last_header_returns_last_matching(self) -> None:
        msg = self.make_valid(
            headers=(("h1", b"a"), ("h2", b"b"), ("h1", b"c")),
        )
        self.assertEqual(msg.last_header("h1"), b"c")
        self.assertEqual(msg.last_header("h2"), b"b")
        self.assertIsNone(msg.last_header("nonexistent"))

    def test_last_header_is_none_for_no_headers(self) -> None:
        msg = self.make_valid(headers=())
        self.assertIsNone(msg.last_header("any"))

    def test_header_values_empty_for_no_headers(self) -> None:
        msg = self.make_valid(headers=())
        self.assertEqual(msg.header_values("any"), ())

    def test_to_dict_encodes_binary_as_base64(self) -> None:
        msg = self.make_valid(
            key=b"key",
            value=b"val",
            headers=(("h", b"hv"),),
            timestamp_ms=42,
            timestamp_type=1,
            leader_epoch=0,
        )
        d = msg.to_dict()
        self.assertEqual(d["topic"], "test-topic")
        self.assertEqual(d["partition"], 0)
        self.assertEqual(d["offset"], 100)
        self.assertEqual(d["key"], "a2V5")
        self.assertEqual(d["value"], "dmFs")
        self.assertEqual(d["headers"], [("h", "aHY=")])
        self.assertEqual(d["timestamp_ms"], 42)
        self.assertEqual(d["timestamp_type"], 1)
        self.assertEqual(d["leader_epoch"], 0)

    def test_to_dict_null_key_encoded_as_none(self) -> None:
        msg = self.make_valid(key=None, value=b"")
        d = msg.to_dict()
        self.assertIsNone(d["key"])
        self.assertEqual(d["value"], "")

    def test_to_dict_none_header_value_encoded_as_none(self) -> None:
        msg = self.make_valid(
            headers=(("h", None),),
        )
        d = msg.to_dict()
        self.assertEqual(d["headers"], [("h", None)])

    def test_to_dict_missing_optionals_are_none(self) -> None:
        msg = ConsumedMessage(
            topic="t",
            partition=0,
            offset=0,
            key=None,
            value=b"v",
        )
        d = msg.to_dict()
        self.assertIsNone(d["timestamp_ms"])
        self.assertIsNone(d["timestamp_type"])
        self.assertIsNone(d["leader_epoch"])


class KafkaMessageProcessingErrorTest(unittest.TestCase):
    def make_message(self, **overrides) -> ConsumedMessage:
        kwargs = dict(
            topic="t", partition=0, offset=1, key=None, value=b"v"
        )
        kwargs.update(overrides)
        return ConsumedMessage(**kwargs)

    def test_creates_with_message(self) -> None:
        msg = self.make_message()
        err = KafkaMessageProcessingError(msg)
        self.assertIs(err.message, msg)

    def test_str_includes_topic_partition_offset(self) -> None:
        msg = self.make_message(topic="my-topic", partition=3, offset=99)
        err = KafkaMessageProcessingError(msg)
        s = str(err)
        self.assertIn("my-topic", s)
        self.assertIn("3", s)
        self.assertIn("99", s)

    def test_is_runtime_error_subclass(self) -> None:
        msg = self.make_message()
        err = KafkaMessageProcessingError(msg)
        self.assertIsInstance(err, RuntimeError)


if __name__ == "__main__":
    unittest.main()
