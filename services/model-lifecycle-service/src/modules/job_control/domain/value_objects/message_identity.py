from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class MessageIdentity:
    topic: str
    partition_id: int
    message_offset: int
    consumer_group: str