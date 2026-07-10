import base64
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Tuple, TypeAlias, Callable
from datetime import datetime, timezone


KafkaHeader: TypeAlias = tuple[str, bytes | None]
KafkaHeaders: TypeAlias = tuple[KafkaHeader, ...]
KafkaMessageIdentity: TypeAlias = tuple[str, int, int]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True,
)
class ConsumedMessage:
    topic: str
    partition: int
    offset: int
    key: bytes | None = field(repr = False)
    value: bytes
    headers: KafkaHeaders = field(
        default = (),
        repr = False,
    )
    timestamp_ms: int | None = None
    timestamp_type: int | None = None
    leader_epoch: int | None = None

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("Kafka topic must not be empty")

        if self.partition < 0:
            raise ValueError(
                f"Kafka partition must be non-negative: {self.partition}"
            )

        if self.offset < 0:
            raise ValueError(
                f"Kafka offset must be non-negative: {self.offset}"
            )

        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError(
                "Kafka timestamp_ms must be non-negative or None"
            )

    @property
    def identity(self) -> KafkaMessageIdentity:
        return self.topic, self.partition, self.offset

    @property
    def next_offset(self) -> int:
        return self.offset + 1
    
    @property
    def value_size_bytes(self) -> int:
        if self.value is None:
            return 0

        return len(self.value)

    @property
    def timestamp(self) -> datetime | None:
        if self.timestamp_ms is None:
            return None

        return datetime.fromtimestamp(
            self.timestamp_ms / 1_000,
            tz=timezone.utc,
        )

    def require_value(self) -> bytes:
        if self.value is None:
            raise ValueError(
                "Kafka message value must not be null: "
                f"topic={self.topic}, "
                f"partition={self.partition}, "
                f"offset={self.offset}"
            )

        return self.value

    def header_values(
        self,
        name: str,
    ) -> tuple[bytes | None, ...]:
        return tuple(
            value
            for header_name, value in self.headers
            if header_name == name
        )

    def last_header(
        self,
        name: str,
    ) -> bytes | None:
        for header_name, value in reversed(self.headers):
            if header_name == name:
                return value

        return None

    def to_dict(self) -> dict:
        def _encode(b: bytes | None) -> str | None:
            if b is None:
                return None
            return base64.b64encode(b).decode("ascii")

        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "key": _encode(self.key),
            "value": _encode(self.value),
            "headers": [
                (name, _encode(val)) for name, val in self.headers
            ],
            "timestamp_ms": self.timestamp_ms,
            "timestamp_type": self.timestamp_type,
            "leader_epoch": self.leader_epoch,
        }


class KafkaMessageProcessingError(RuntimeError):
    def __init__(self, message: ConsumedMessage) -> None:
        self.message = message

        super().__init__(
            "Failed to process Kafka message "
            f"topic={message.topic}, "
            f"partition={message.partition}, "
            f"offset={message.offset}"
        )

MessageHandler: TypeAlias = Callable[[ConsumedMessage], None]