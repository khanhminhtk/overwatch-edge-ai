from abc import ABC, abstractmethod

class BaseCTCLabelEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass