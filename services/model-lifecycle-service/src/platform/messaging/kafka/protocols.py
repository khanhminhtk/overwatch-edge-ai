from typing import Protocol, runtime_checkable


@runtime_checkable
class MessageProducer(Protocol):
    def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes | None,
        headers: list[tuple[str, bytes | None]] | None = None,
    ) -> None: ...


@runtime_checkable
class MessageConsumer(Protocol):
    def run(self) -> None: ...
    def stop(self) -> None: ...


class ConfluentConsumer(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...


class ConfluentProducer(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...
