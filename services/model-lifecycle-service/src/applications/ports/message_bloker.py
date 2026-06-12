from abc import ABC, abstractmethod

class IMessageBlokerClient(ABC):
    @abstractmethod
    def block_message(self, message_id: str) -> None:
        pass

    @abstractmethod
    def unblock_message(self, message_id: str) -> None:
        pass

    @abstractmethod
    def is_message_blocked(self, message_id: str) -> bool:
        pass

class IMessageBlokerConsummer(ABC):
    @abstractmethod
    async def consume_message(self, message_id: str) -> None:
        pass

class IMessageBlokerProducer(ABC):
    @abstractmethod
    async def produce_message(self, message_id: str) -> None:
        pass